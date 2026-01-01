import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score

import numpy as np
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_embedding(embedding_model_name, sentences):
    # sentences: list with length n
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(sentences)
    # embeddings shape: numpy array [n, 384]
    return embeddings

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    
    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # method-specific hyperparameters
    embedding_model_name = hyperparameters.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    hidden_features = hyperparameters.get("hidden_features", 8)
    in_edges = hyperparameters.get("in_edges", 3)  # cost and effect
    learning_rate = hyperparameters.get("learning_rate", 1e-4)
    weight_decay = hyperparameters.get("weight_decay", 1e-4)
    train_epochs = hyperparameters.get("train_epochs", 50)
    batch_size = hyperparameters.get("batch_size", 32)
    train_mask_rate = hyperparameters.get("train_mask_rate", 0.5)
    split_ratio = hyperparameters.get("split_ratio", [0.7, 0.15, 0.15])  # train, val, test
    scenario = hyperparameters.get("scenario", "Performance First")  # "Performance First", "Balance", "Cost First"
    model_descriptions = hyperparameters.get("model_descriptions", None)
    task_description = hyperparameters.get("task_description", None) # default to be task name
    
    assert model_descriptions != None, "The model_descriptions is needed in hyperparameters in task config."
    assert task_description != None, "The task_description is needed in hyperparameters in task config."
    assert len(model_descriptions) == len(model_names), "The number of model_descriptions must match model numbers."

    # Preparing router training data from dev set and get scores
    print("Preparing dev set data...")
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    list_of_input_list = [dev_input_list for _ in model_names]
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids,
    )

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_score) / len(dev_score)
        list_of_dev_scores.append(dev_score)
        print("Model: {}, dev {} score: {}".format(model_names[i], task_type, avg_dev_score))

    # Extract embeddings for models (use descriptions if available, otherwise model names)
    print("Extracting model embeddings...")
    model_texts = model_descriptions
    model_embeddings = get_embedding(embedding_model_name, model_texts) # shape [number_model, 384]

    # Extract embeddings for queries
    print("Extracting query embeddings...")
    query_embeddings = get_embedding(embedding_model_name, dev_input_list) # shape [number_query, 384]
    
    # Prepare graph data structure
    print("Preparing graph data structure...")
    num_queries = len(dev_input_list)
    num_models = len(model_names)
    
    # Create edge features: [effect (score), cost (normalized, can be 1.0 for now)]
    effect_list = []
    cost_list = []
    for j in range(num_queries):
        for i in range(num_models):
            score = float(list_of_dev_scores[i][j])
            effect_list.append(score) # [number_query * number_model, ]
            cost_list.append(1.0)  # Can be customized based on model cost
    
    effect_list = np.array(effect_list) # [number_query * number_model, ]
    cost_list = np.array(cost_list) # [number_query * number_model, ]
    
    # Apply scenario-based weighting
    if scenario == "Performance First":
        combined_effect = 1.0 * effect_list - 0.0 * cost_list # [number_query * number_model, ]
    elif scenario == "Balance":
        combined_effect = 0.5 * effect_list - 0.5 * cost_list
    else:  # Cost First
        combined_effect = 0.2 * effect_list - 0.8 * cost_list
    
    # Create labels (binary indicator: 1 if this edge is the best model for the query, 0 otherwise)
    # For each query, find the best model, then create binary labels for all edges
    effect_reshaped = combined_effect.reshape(-1, num_models)  # (num_queries, num_models)
    best_model_per_query = np.argmax(effect_reshaped, axis=1)  # (num_queries,)
    labels = np.zeros(num_queries * num_models)  # Binary labels for each edge, (num_queries * num_models, )
    for q_idx in range(num_queries):
        best_model_idx = best_model_per_query[q_idx]
        edge_idx = q_idx * num_models + best_model_idx
        labels[edge_idx] = 1.0
    
    # Create edge indices: query nodes -> model nodes
    # Query nodes: 0 to num_queries-1
    # Model nodes: num_queries to num_queries+num_models-1
    edge_org_id = [num for num in range(num_queries) for _ in range(num_models)]
    # des_node pattern matches graph_router_multi_task.py: [0, 1, ..., num_models-1] repeated num_queries times
    edge_des_id = list(range(num_models)) * num_queries
    
    # Split data into train/val/test
    split_ratio = hyperparameters.get("split_ratio", [0.8, 0.1, 0.1])
    train_size = int(num_queries * split_ratio[0])
    val_size = int(num_queries * split_ratio[1])
    test_size = num_queries - train_size - val_size
    
    # Create masks for edges
    train_indices = list(range(train_size * num_models))
    val_indices = list(range(train_size * num_models, (train_size + val_size) * num_models))
    test_indices = list(range((train_size + val_size) * num_models, num_queries * num_models))
    
    mask_train = torch.zeros(len(edge_org_id))
    mask_train[train_indices] = 1
    
    mask_validate = torch.zeros(len(edge_org_id))
    mask_validate[val_indices] = 1
    
    mask_test = torch.zeros(len(edge_org_id))
    mask_test[test_indices] = 1
    
    # Create task embeddings (can use task name or description)
    task_embeddings = get_embedding(embedding_model_name, [task_description] * num_queries)
    
    # Prepare combined edge features, [number_query * number_model, 2]
    combined_edge = np.concatenate((cost_list.reshape(-1, 1), effect_list.reshape(-1, 1)), axis=1)
    
    # Form graph data
    device = "cuda:{}".format(gpu_ids[0])
    form_data_obj = form_data(device)
    
    query_dim = query_embeddings.shape[1] # 384
    model_dim = model_embeddings.shape[1] # 384
    task_dim = task_embeddings.shape[1] # 384
    
    # Prepare data for GNN - use full task_embeddings list (one per query)
    data_train = form_data_obj.formulation(
        task_id=task_embeddings,
        query_feature=query_embeddings,
        llm_feature=model_embeddings,
        org_node=edge_org_id,
        des_node=edge_des_id,
        edge_feature=effect_list,
        edge_mask=mask_train,
        label=labels,
        combined_edge=combined_edge,
        train_mask=mask_train,
        valide_mask=mask_validate,
        test_mask=mask_test
    )
    
    data_validate = form_data_obj.formulation(
        task_id=task_embeddings,
        query_feature=query_embeddings,
        llm_feature=model_embeddings,
        org_node=edge_org_id,
        des_node=edge_des_id,
        edge_feature=effect_list,
        edge_mask=mask_validate,
        label=labels,
        combined_edge=combined_edge,
        train_mask=mask_train,
        valide_mask=mask_validate,
        test_mask=mask_test
    )
    
    data_test = form_data_obj.formulation(
        task_id=task_embeddings,
        query_feature=query_embeddings,
        llm_feature=model_embeddings,
        org_node=edge_org_id,
        des_node=edge_des_id,
        edge_feature=effect_list,
        edge_mask=mask_test,
        label=labels,
        combined_edge=combined_edge,
        train_mask=mask_train,
        valide_mask=mask_validate,
        test_mask=mask_test
    )
    
    # Train GNN router
    print("Training GNN router...")
    model_path = f"model_collaboration/logs/graph_router_{task}_{len(model_names)}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    config = {
        'embedding_dim': hidden_features,
        'edge_dim': in_edges,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'train_epoch': train_epochs,
        'batch_size': batch_size,
        'train_mask_rate': train_mask_rate,
        'llm_num': num_models,
        'model_path': model_path
    }
    
    # Create a dummy wandb object if not available
    class DummyWandb:
        def log(self, *args, **kwargs):
            pass

    gnn_predict = GNN_prediction(
        query_feature_dim=query_dim,
        llm_feature_dim=model_dim,
        hidden_features_size=hidden_features,
        in_edges_size=in_edges,
        wandb=DummyWandb(),
        config=config,
        device=device
    )
    
    gnn_predict.train_validate(data_train, data_validate, data_test)
    
    # Load the trained model for inference
    gnn_predict.model.load_state_dict(torch.load(model_path))
    gnn_predict.model.eval()
    
    # Using the router for the test set
    print("Routing test set queries...")
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    
    # Extract embeddings for test queries
    test_query_embeddings = get_embedding(embedding_model_name, test_input_list)
    
    # Create test task embeddings (ensure same dimension as model embeddings)
    test_task_embeddings = get_embedding(embedding_model_name,[task_description] * len(test_input_list))
    
    # Prepare test graph data
    num_test_queries = len(test_input_list)
    test_edge_org_id = [num for num in range(num_test_queries) for _ in range(num_models)]
    # des_node pattern: [0, 1, ..., num_models-1] repeated num_test_queries times
    test_edge_des_id = list(range(num_models)) * num_test_queries
    
    # Create dummy edge features and labels for test (not used in prediction)
    test_effect_list = np.ones(num_test_queries * num_models)
    test_cost_list = np.ones(num_test_queries * num_models)
    test_combined_edge = np.concatenate((test_cost_list.reshape(-1, 1), test_effect_list.reshape(-1, 1)), axis=1)
    test_labels = np.zeros(num_test_queries * num_models)  # Dummy binary labels (not used in prediction)
    
    test_mask = torch.ones(num_test_queries * num_models)
    # For test inference, edge_can_see should be empty since we haven't seen these test edges during training
    # The model will use the learned node representations to make predictions
    train_mask_for_test = torch.zeros(num_test_queries * num_models)  # No training edges in test set
    valide_mask_for_test = torch.zeros(num_test_queries * num_models)  # No validation edges in test set
    
    data_test_inference = form_data_obj.formulation(
        task_id=test_task_embeddings,
        query_feature=test_query_embeddings,
        llm_feature=model_embeddings,
        org_node=test_edge_org_id,
        des_node=test_edge_des_id,
        edge_feature=test_effect_list,
        edge_mask=test_mask,
        label=test_labels,
        combined_edge=test_combined_edge,
        train_mask=train_mask_for_test,
        valide_mask=valide_mask_for_test,
        test_mask=test_mask
    )
    
    # Predict best model for each test query
    selected_model_indices = []
    with torch.no_grad():
        # For test inference on a new graph, we use all test edges for the encoder
        # (since we haven't seen any of them during training, we use the test graph structure itself)
        # This allows the model to perform graph convolution on the test graph
        edge_can_see = torch.ones(num_test_queries * num_models, dtype=torch.bool)
        
        edge_predict = gnn_predict.model(
            task_id=data_test_inference.task_id,
            query_features=data_test_inference.query_features,
            llm_features=data_test_inference.llm_features,
            edge_index=data_test_inference.edge_index,
            edge_mask=test_mask.bool(),
            edge_can_see=edge_can_see,
            edge_weight=data_test_inference.combined_edge
        )
        
        # Reshape predictions and select best model for each query
        edge_predict_reshaped = edge_predict.reshape(-1, num_models)
        selected_model_indices = torch.argmax(edge_predict_reshaped, dim=1).cpu().numpy().tolist()
    
    assert max(selected_model_indices) < len(model_names), "Selected model index out of range"
    assert min(selected_model_indices) >= 0, "Selected model index out of range"
    
    # Generate final outputs
    print("Generating final outputs...")
    list_of_input_list = []
    for i in range(len(model_names)):
        model_input_list = []
        for j in range(len(test_input_list)):
            if selected_model_indices[j] == i:
                model_input_list.append(test_input_list[j])
        list_of_input_list.append(model_input_list)
    
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids,
    )
    
    final_outputs = []
    for j in range(len(test_input_list)):
        model_idx = selected_model_indices[j]
        final_outputs.append(list_of_output_list[model_idx][0])
        list_of_output_list[model_idx].pop(0)
    
    # Evaluate
    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score after graph router: {}".format(task, avg_test_score))
    
    # Save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "selected_model": model_names[selected_model_indices[i]],
            "output": final_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)
    
    log_filename = "model_collaboration/logs/{}_{}_{}_graph_router.json".format(task, len(model_names), round(avg_test_score, 4))
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)
    
    # Cleanup
    del gnn_predict
    torch.cuda.empty_cache()
    
    return 0

class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim) # 384, 8
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim*2) # 384, 16
        self.task_transform = nn.Linear(llm_feature_dim, common_dim) # 384, 8

    def forward(self, task_id, query_features, llm_features):
        aligned_task_features = self.task_transform(task_id) # 8
        aligned_query_features = self.query_transform(query_features)  # 8
        aligned_two_features=torch.cat([aligned_task_features,aligned_query_features], 1) # 16
        aligned_llm_features = self.llm_transform(llm_features) # 16
        aligned_features = torch.cat([aligned_two_features, aligned_llm_features], 0) # bs, 16
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features * 2)
        self.bn2 = nn.BatchNorm1d(hidden_features * 2)

    def forward(self, task_id, query_features, llm_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):
        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]
        edge_weight_mask = F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1,self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features))
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        edge_predict = F.sigmoid(
            (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1))
        return edge_predict

class form_data:

    def __init__(self,device):
        self.device = device

    def formulation(self,task_id,query_feature,llm_feature,org_node,des_node,edge_feature,label,edge_mask,combined_edge,train_mask,valide_mask,test_mask):

        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        task_id = torch.tensor(task_id, dtype=torch.float).to(self.device)
        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]
        des_node=[(i+1 + org_node[-1]) for i in des_node]
        edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(self.device)
        edge_weight = torch.tensor(edge_feature, dtype=torch.float).reshape(-1,1).to(self.device)
        combined_edge = torch.tensor(combined_edge, dtype=torch.float).reshape(-1,2).to(self.device)
        combined_edge=torch.cat((edge_weight, combined_edge), dim=-1)
        data = Data(task_id=task_id,query_features=query_features, llm_features=llm_features, edge_index=edge_index,
                        edge_attr=edge_weight,query_indices=query_indices, llm_indices=llm_indices,label=torch.tensor(label, dtype=torch.float).to(self.device),
                        edge_mask=edge_mask,combined_edge=combined_edge,
                    train_mask=train_mask,valide_mask=valide_mask,test_mask=test_mask)

        return data

class GNN_prediction:

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features_size, in_edges_size, wandb, config, device):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim, 
                                       llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,
                                        in_edges=in_edges_size).to(device)
        self.wandb = wandb
        self.config = config
        self.optimizer =AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.criterion = torch.nn.BCELoss()

    def train_validate(self, data, data_validate, data_for_test):

        best_f1 = -1
        self.save_path= self.config['model_path']
        self.num_edges = len(data.edge_attr)
        self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)
        for epoch in range(self.config['train_epoch']):
            self.model.train()
            loss_mean = 0
            mask_train = data.edge_mask
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone()
                mask = mask.bool()
                random_mask = torch.rand(mask.size()) < self.config['train_mask_rate']
                random_mask = random_mask.to(torch.bool)
                mask = torch.where(mask & random_mask, torch.tensor(False, dtype=torch.bool), mask)
                mask = mask.bool()
                edge_can_see = torch.logical_and(~mask, self.train_mask)
                self.optimizer.zero_grad()
                predicted_edges= self.model(task_id=data.task_id,
                                            query_features=data.query_features, 
                                            llm_features=data.llm_features, 
                                            edge_index=data.edge_index,
                                            edge_mask=mask,
                                            edge_can_see=edge_can_see,
                                            edge_weight=data.combined_edge)
                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean += loss
            loss_mean=loss_mean / self.config['batch_size']
            loss_mean.backward()
            self.optimizer.step()

            self.model.eval()
            mask_validate = torch.tensor(data_validate.edge_mask, dtype=torch.bool)
            edge_can_see = self.train_mask
            with torch.no_grad():
                predicted_edges_validate = self.model(task_id=data_validate.task_id,
                                                      query_features=data_validate.query_features,
                                                      llm_features=data_validate.llm_features,
                                                      edge_index=data_validate.edge_index,
                                                      edge_mask=mask_validate,
                                                      edge_can_see=edge_can_see, 
                                                      edge_weight=data_validate.combined_edge)
                observe_edge = predicted_edges_validate.reshape(-1, self.config['llm_num'])
                observe_idx = torch.argmax(observe_edge, 1)
                value_validate = data_validate.edge_attr[mask_validate].reshape(-1, self.config['llm_num'])
                label_idx = torch.argmax(value_validate, 1)
                correct = (observe_idx == label_idx).sum().item()
                total = label_idx.size(0)
                validate_accuracy = correct / total
                observe_idx_ = observe_idx.cpu().numpy()
                label_idx_ = label_idx.cpu().numpy()
                # calculate macro F1 score
                f1 = f1_score(label_idx_, observe_idx_, average='macro')
                loss_validate = self.criterion(predicted_edges_validate.reshape(-1), data_validate.label[mask_validate].reshape(-1))

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(self.model.state_dict(), self.save_path)
                test_result,test_loss=self.test(data_for_test,self.config['model_path'])
                self.wandb.log({"train_loss": loss_mean, "validate_loss": loss_validate, "test_loss": test_loss, "validate_accuracy": validate_accuracy, "validate_f1": f1, "test_result": test_result})

    def test(self, data, model_path):
        # self.model.load_state_dict(model_path)
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,
                                      query_features=data.query_features, 
                                      llm_features=data.llm_features, 
                                      edge_index=data.edge_index,
                                      edge_mask=mask,
                                      edge_can_see=edge_can_see,
                                      edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, 1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, 1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        print("result_predict:", result, "result_golden:",result_golden)

        return result, loss_test

if __name__ == "__main__":
    run_method()


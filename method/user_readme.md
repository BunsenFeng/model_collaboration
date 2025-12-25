### General
Use JSON config files to specify the method and hyperparameters. Use `python main.py -c your_config.json` to run the experiments.

Config files look like:
```json
{
    "method": "api_prompt_routing", // the name under method/ folder, names like <type>_<approach>.py
    "task": "agieval", // the name under data/ folder, see data/eval_readme.md
    "task_type": "multiple_choice", // see data/eval_readme.md
    "gpu_ids": [0,1,2], // a list of GPUs available
    "model_names": [
        "model_1_name",
        "model_2_name",
        "model_3_name"
    ], // a list of model identifiers, local or huggingface
    "hyperparameters": {
        "max_response_length": 512, // max generation length
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 8, // per GPU batch size
        // and then, method-specific hyperparameters
    }
}
```

len(gpu_ids) can be fewer than len(model_names) in most approaches. But please, try to use multiple GPUs and ideally len(gpu_ids) == len(model_names). The code will automatically assign models to GPUs in a round-robin manner. Use <10B LMs. These are vibe implementations (and your future implementations will be): they are not meant to reproduce every single niche detail in any paper, just taking the core ideas and making them work in a reasonable way.

If you are trying to run collaboration with one of the model being too large to fit onto a single GPU: add `"big_model_mode": true` to `"hyperparameters"`: it will use all provided GPUs for a single model in rotation. This will only work for some approaches.

Reasoning LMs are supported! Please use much larger `"max_response_length"` to account for them: we will parse the text after `</think>` as the actual model output.

#### API-level: Nudging
- file: `api_nudging.py`
- description: A training-free guided decoding method. During generation, if the base model is uncertain about the next token (top-1 prob < `gamma`), a (smaller) nudging model intervenes. The nudging model inserts nudging token(s) (often stylistic/discourse markers) to guide the base model’s generation. 
    - Implementation Details: The method ensures complete words are inserted (by splitting on spaces) to maintain coherence and support collaboration of models with different tokenizers. Generation stops when either the base or nudging model emits an EOS token.
- related paper(s):
    - [Nudging: Inference-time Alignment of LLMs via Guided Decoding](https://arxiv.org/abs/2410.09300)
- method-specific hyperparameters:
    - `gamma` (float, default 0.4): The uncertainty threshold. If the base model's top-1 probability is below this value, the nudging model takes over.
    - `base_model_id` (int, default 0): the index of the base model in `model_names`.
    - `nudging_model_id` (int, default 1): the index of the nudging model in `model_names`. (Requires `len(model_names) >= 2`). All the models that are not the base model are potential nudging models and will be used for searching if `search_nudging` is True.
    - `search_gamma` (bool, default False): If `True`, performs a grid search over gamma values `[0.2, 0.3, 0.4, 0.5]` using the dev set to find the optimal threshold.
    - `search_nudging` (bool, default False): If `True`, search over all potential nudging models (all the models in `model_names` excluding `base_model_id`) using the dev set to find the best nudging model.
- performance & usage note: 
    - Model support: Supports models from different families (tested on `Llama-3.1-Tulu-3-8B` + `Gemma-2-2b-it` / `Llama-3.1-Tulu-3-8B-DPO`). 
    - Inference speed: This method is currently implemented without KV-caching (stateless inference). Consequently, batched inference is compute-bound rather than memory-bound.
        - Expectation: Speed scales linearly with batch size. A batch of size $N$ will take roughly $N$ times longer than the longest sample in the batch.
        - For minimizing the inference time, one should consider setting `batch_size` to be small or group samples by length.

#### API-level: Prompt Routing
- file: `api_prompt_routing.py`
- description: prompt an LLM to route among the candidate LLMs based on their descriptions. First, evaluate models on the dev set to determine who is best and use it for the routing. Then, given (model descriptions, query), this LLM decides which candidate model (including itself) should be selected. Finally, generation with the selected LLM for each query.
- method-specific hyperparameters:
    - `model_descriptions`: a list of strings describing each candidate model, in the same order as `model_names`.
- note to tester: just try different LLMs you'd like.

#### API-level: Switch Generation
- file: `api_switch_generation.py`
- description: train (or use an existing LM) for switching among candidate LLMs (choosing different models to generate a patch of text before switching to another model). First, train a switcher LM based on rollouts evaluated by a reward model or use an existing switcher LM checkpoint. Then, switch generation among candidate LLMs based on the switcher LM's guidance.
- related paper(s):
    - [Don't Throw Away Your Pretrained Model](https://arxiv.org/abs/2510.09913)
- method-specific hyperparameters:
    - `patch_size`, default 25: the number of tokens generated by each candidate model before switching.
    - `selector_model_name`, default None: if provided, load the switcher LM from this checkpoint; otherwise, train a new switcher LM on this task. You can try `bunsenfeng/PFA_switcher_1`, `bunsenfeng/PFA_switcher_2`, or any instruction following model.
    - `selector_base_model`, default `Qwen/Qwen2.5-7B-Instruct`: the initial model for training the switcher LM if `selector_model_name` is not provided.
    - `objective_flag`, default False: whether to solicit a definitive answer in the last patch (by adding "The answer is") to the last patch prompt.
    - `training_instance_num`, default 1000: the number of training instances to use for training the switcher LM.
    - `rollout_per_instance`, default 16: the number of rollouts to generate for each training instance.
    - `reward_model_gpu_id`, default None: the GPU ID to load the reward model; if None, use the first GPU in `gpu_ids`.
    - `reward_model_name`, default `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`: the reward model to evaluate the rollouts.
    - `wait_flag`, default True: whether to add a "Wait" to patches with fewer than 20 characters (to avoid ending so soon), in an s1 TTS fashion.
- warning: cost could be high with `selector_model_name` not provided and a large `training_instance_num` and `rollout_per_instance`. Use a selector model or reduce the number of training instances and rollouts to save cost.
- note to tester: try one `selector_model_name` among `bunsenfeng/PFA_switcher_1`, `bunsenfeng/PFA_switcher_2`, or any instruction following model. Then try another one with `selector_model_name` none to train a new switcher LM on the task.

#### API-level: Trained Router
- file: `api_trained_router.py`
- description: train a router model to select among candidate LLMs. First, evaluate models on the dev set to determine who is best and use it for training the router. (if a tie on the task, for example when both generations are correct, use a reward model as the tie breaker) Then, given (model descriptions (optional), query), train a router model to select the best candidate model. Finally, generation with the selected LLM for each query.
- related paper(s):
    - [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)
- method-specific hyperparameters:
    - `router_base_model`, default `Qwen/Qwen2.5-7B-Instruct`: the initial model for training the router.
    - `model_descriptions`, default None: a list of strings describing each candidate model, in the same order as `model_names`. Optional. If provided, the router model input will include model descriptions.
    - `reward_model_gpu_id`, default `gpu_ids[0]`: the GPU ID to load the reward model.
    - `reward_model_name`, default `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`: the reward model to evaluate generations when there is a tie on the task.
- note to tester: try different LLMs you'd like.

#### API-level: Graph Router
- file: `api_graph_routing.py`
- description: train a graph router model to select among candidate LLMs. First, evaluate models on the dev set to get scores list to determine who is best on each query. Secondly, construct graph router train-validate input data and train graph model. Thirdly, generation with the selected LLM for each query.
- related paper(s):
    - [GraphRouter: A Graph-based Router for LLM Selections](https://arxiv.org/pdf/2410.03834)
- method-specific hyperparameters:
    - `embedding_model`, default `sentence-transformers/all-MiniLM-L6-v2`: the model for extracting embedding.
    - `model_descriptions`, default None: a list of strings describing each candidate model, in the same order as `model_names`. 
    - `task_description`, default task name: a string describing task. 
    - `hidden_features`, default 8: the hidden dimension for graph router.
    - `in_edges`, default 3: the input features number for graph router.
    - `train_mask_rate`, default 0.5: the rate to mask train data.
    - `scenario`, default `Performance First`: the balance between performance and cost.
- note to tester: try different LLMs you'd like.

#### API-level: Cascade
- file: `api_cascade.py`
- description: given a list of models from weak to strong, defer question to next model if current model is unconfident.
- related paper(s):
    - [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/pdf/2305.05176)
    - [Language Model Cascades: Token-level uncertainty and beyond](https://arxiv.org/pdf/2404.10136)
- method-specific hyperparameters:
    - `mode`, default `logit`: the mode for judge model confidence. Mode `logit` average token probability in response as model confidence. If confidence less than threshold, then deferred. Mode `just_ask` ask model to mention it is unconfident of its answer. If unconfident is in output, then deferred.
    - `percentage`, default 0.5: the percentage to select deferral threshold in `logit` mode. For each model except last model, calculate responses' scores on dev set. For each model, select its threshold as top `percentage` value in its dev set responses scores.
    - (suggested) `model_names`: arrange model names from weak to strong, from cheap to expensive. For instance, `["Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-7B-Instruct"]`
- note to tester: try different LLMs you'd like.

#### API-level: MentorCollab
- file: `text_mentor_collab.py`
- description: collaborative generation between a generator model (typically a small model) and a mentor model (typically a large reasoning model). During generation, both models operate in parallel. At each decision point, if the next predicted token differs between the two models, both generate a segment of text (patch). The method then decides which segment to follow either through: (1) "free" mode where the generator model itself judges which option is better, or (2) "train" mode where a trained MLP classifier predicts the better choice based on the generator's hidden states. This allows the base model to selectively incorporate guidance from the instruction-tuned mentor throughout generation.
- method-specific hyperparameters:
    - `decision_proportion`, default 25: percentage (0-100) of generation steps where the mentor is consulted. At other steps, the generator proceeds independently.
    - `patch_size`, default 16: number of tokens to generate in each segment when both models' predictions diverge.
    - `mode`, default "free": decision strategy. Options are "free" (generator self-judges) or "train" (use trained MLP classifier).
    - `task`, default "General": task type for loading the trained MLP model. Options are "Math" or "General" (only used in "train" mode).
    - `mlp_threshold`, default 0.5: decision threshold for the MLP classifier. Scores above this threshold choose the generator's segment, otherwise choose the mentor's (only used in "train" mode).
- notes:
    - requires exactly 2 models and 2 GPUs. The first model should be a base model (generator), and the second should be an instruction-tuned variant (mentor). In "train" mode, only specific model pairs are supported (see `MENTOR_COLLAB_TRAIN_SUPPORT_MODELS` in the code).
    - try `["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-14B"]` with `mode: "free"` first. For "train" mode, ensure the generator model is in the supported list. 
    - supported generator list for "train" mode: `["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B-Base", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-4b-it", "google/gemma-3-4b-pt"]`

#### Text-level: Multiagent Refine/Debate
- file: `text_multiagent_refine.py`
- description: multiple LLMs collaborate to refine the answers of each other. First, evaluate all models on the dev set to select a final summarizer. At each round, each LLM sees the answers of all LLMs from the previous round and refines its own answer. After several rounds, the final answers are aggregated by the summarizer LLM.
- related paper(s):
    - [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
    - [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)
- method-specific hyperparameters:
    - `round`, default 3: the number of refinement rounds.
- note to tester: just try different LLMs you'd like.

#### Text-level: Multiagent Feedback
- file: `text_multiagent_feedback.py`
- description: multiple LLMs collaborate by providing feedback to each other. First, evaluate all models on the dev set to select a final summarizer. For each query, each LLM generates an initial answer, then provides feedback to other LLMs' answers, and finally refines its answer based on the received feedback. After several rounds, the final answers are aggregated by the summarizer LLM.
- related paper(s):
    - [Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration](https://arxiv.org/abs/2402.00367)
- method-specific hyperparameters:
    - `round`, default 3: the number of feedback rounds.
    - `feedback_count`, default 3: how many other LLMs to provide feedback to each LLM in each round.
- note to tester: just try different LLMs you'd like.

#### Text-level: Knowledge Card
- file: `text_knowledge_card.py`
- description: augment one LLM with the knowledge of other LLMs. First, evaluate all models on the dev set to select a final response generator LLM. For each query, each LLM (including or excluding self) generates a paragraph of knowledge and information relevant to the query. The final LLM then generates the final answer based on the aggregated knowledge from all LLMs.
- related paper(s):
    - [Knowledge Card: Filling LLMs' Knowledge Gaps with Plug-in Specialized Language Models](https://arxiv.org/abs/2305.09955)
    - [Generated Knowledge Prompting for Commonsense Reasoning](https://arxiv.org/abs/2110.08387)
- method-specific hyperparameters:
    - `exclude_self`, default True: whether to exclude the response generator LLM when generating knowledge.
- note to tester: just try different LLMs you'd like.

#### Text-level: LLM Blender
- file: `text_llm_blender.py`
- description: implement a text-level version of LLM-Blender. For each query, all base LLMs first generate their own candidate answers. A ranker LLM, optionally fine-tuned on dev pairwise preferences, compares candidates in an A/B fashion and aggregates wins to obtain a ranking. The top-k ranked candidates are then passed to a fuser LLM, optionally fine-tuned on dev with gold answers as supervision (when available), which generates a single fused answer that combines and refines the candidates. The same pipeline is applied on the test set for evaluation.
- related paper(s):
    - [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generation Fusion](https://arxiv.org/abs/2402.08115)
- method-specific hyperparameters:
    - `top_k`, default 3: how many top-ranked candidate answers to pass from the ranker to the fuser for each query. Will be clipped to `[1, len(model_names)]`.
    - `ranker_model_name`, default `Qwen/Qwen2.5-7B-Instruct`: causal LM (HuggingFace name or local path) used as the ranker/judge model. Given `(question, candidates)`, this model is prompted to output scores for each candidate.
    - `ranker_gpu_id`, default `gpu_ids[0]`: GPU id to use for ranker inference.
    - `ranker_max_response_length`, default 128: maximum number of tokens to generate when the ranker LM is judging candidate answers (short outputs are sufficient).
    - `fuser_model_name`, default first entry of `model_names`: causal LM used as the fuser model that reads the question and top-k candidate answers and generates the final answer.
    - `fuser_gpu_id`, default `gpu_ids[0]`: GPU id to use for fuser inference.
    - `fuser_max_response_length`, default `max_response_length` from the config: maximum number of tokens for the fused final answer.
    - `train_ranker_on_dev`, default False: whether to train the ranker on the dev set before evaluation on the test set. If True, the method will (1) generate candidate answers and scores on the dev split for all base models, (2) construct pairwise preference data based on dev scores, and (3) run SFT via `utils.distributed_sft.single_sft` to fine-tune `ranker_model_name`. At inference time, the ranker is always used in a pairwise A/B comparison mode with win-count aggregation. The trained checkpoint is saved under `logs/text_llm_blender/ranker_model_<task>_<num_models>/` and used only for the current run.
    - `train_fuser_on_dev`, default False: whether to train the fuser on the dev set before evaluation on the test set. If True, the method will (1) use dev scores to select top-k candidate answers per example, (2) build fusion prompts with these candidates, (3) use the dataset gold dev answer as the supervision target when available (and fall back to the best-scoring candidate otherwise), and (4) run SFT via `utils.distributed_sft.single_sft` to fine-tune `fuser_model_name`. The trained checkpoint is saved under `logs/text_llm_blender/fuser_model_<task>_<num_models>/` and used only for the current run.
    - `ranker_sft_batch_size`, default 1: per-device train batch size for ranker SFT when `train_ranker_on_dev` is True.
    - `ranker_sft_gradient_accumulation_steps`, default 16: gradient accumulation steps for ranker SFT.
    - `ranker_sft_learning_rate`, default 1e-5: learning rate for ranker SFT.
    - `ranker_sft_epoch`, default 3: number of training epochs for ranker SFT.
    - `fuser_sft_batch_size`, default 1: per-device train batch size for fuser SFT when `train_fuser_on_dev` is True.
    - `fuser_sft_gradient_accumulation_steps`, default 16: gradient accumulation steps for fuser SFT.
    - `fuser_sft_learning_rate`, default 1e-5: learning rate for fuser SFT.
    - `fuser_sft_epoch`, default 3: number of training epochs for fuser SFT.
- note to tester:
    - For pure zero-shot ranking and fusion, set `train_ranker_on_dev: false` and `train_fuser_on_dev: false`, and choose reasonable `ranker_model_name` / `fuser_model_name` (e.g., `ranker_model_name: "Qwen/Qwen2.5-7B-Instruct"`, `fuser_model_name` as one of your candidate models).
    - To adapt the ranker and/or fuser to a specific task, set the corresponding `train_*_on_dev` flags to true. Intermediate SFT data and checkpoints will be saved under `logs/text_llm_blender/` (e.g., `ranker_sft_<task>_<num_models>.jsonl`, `ranker_model_<task>_<num_models>/`). The final evaluation log remains `logs/<task>_<num_models>_<score>_llm_blender.json` as usual.

#### Text-level: Heterogeneous Swarms
- file: `text_heterogeneous_swarms.py`
- description: multiple LLMs form a directed acyclic graph structure to collaboratively generate responses. Each LLM's output is passed to other LLMs through the directed edges in the graph, to become part of the input context of another LLM. The graph structure is optimized with particle swarm optimization on the dev set to maximize performance.
- related paper(s):
    - [Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](https://arxiv.org/abs/2502.04510)
    - [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)
- method-specific hyperparameters:
    - `population`, default 5: the population size for particle swarm optimization, essentially how many graph structures to explore in each iteration.
    - `max_iterations`, default 5: the maximum number of iterations for particle swarm optimization.
    - There are more hyperparameters for particle swarm optimization, please refer to `text_heterogeneous_swarms.py`. Only change them if you know what you are doing.
- warning: this could be slow with large `population` and `max_iterations`. Reduce them to save computation.

#### Text-level: Majority Vote
- file: `text_majority_vote.py`
- description: Multiple LLMs independently generate answers for each query. The final answer is determined through majority voting, where the answer appearing most frequently among the models is selected as the output. When there is a tie, the tie-breaking strategy is used to select the final answer. This approach is applicable only to question types of "multiple_choice", "exact_match", or "f1_match".
- method-specific hyperparameters:
    - `tie`, default "random": the tie-breaking strategy. Options are "random" (arbitrarily select one of the tied answers) or "dev-based" (evaluate the models that vote for tied answers on the dev set, then use the answer from the best-performing model).

#### Text-level: Structured Interaction
- file: `text_structure.py`
- description: multiple LLMs interact and update their responses according to a specific communication topology (graph structure). First, all models generate initial responses to the query. Then, over multiple rounds, each LLM receives the most recent responses of its "neighboring" models (defined by the structure) as context to update its own answer. Finally, the responses from the last round are evaluated. This allows simulating different information flow patterns like stars, trees, or chains.
- related paper(s):
    - [NetSafe: Exploring the Topological Safety of Multi-agent System](https://aclanthology.org/2025.findings-acl.150/)
- method-specific hyperparameters:
    - `num_rounds`, default 3: the number of response rounds for each model. The first round is when all model generate their initial responses individually, and then they update the responses in the following (`num_rounds` - 1) rounds.
    - `structure_type`, required: the topology of the communication graph. Options are `chain` (linear), `tree` (hierarchical), `star` (centralized), `circle` (ring), `complete` (all-to-all), or `other` (custom, where `structure_matrix` is expected).
    - `structure_matrix`, default None: an $N*N$ adjacency matrix $M$ (list of lists containing 0s and 1s). Required only if `structure_type` is `other`. $M_{ij}=1$ means Model $j$ sees responses by Model $i$ when updating its responses. The indices $i$ and $j$ correspond to the sequence of models defined in `model_names`.
- note: 
    - In each round, the model only sees the outputs from the immediately preceding round, not the entire conversation history.
    - The `structure_matrix` initialized for built-in options is bi-directional by default. However, when using `other`, the matrix does not have to be bi-directional and allows for custom uni-directional flows.
    - If a model does not have in-edges according to the `structure_matrix`, then it will keep its output from the last turn.
    - The models are evaluated on the dev set first without any interaction, where the best model will be selected to compute the final score on the test set after collaboration.
#### Text-level: Multiagent Finetuning
- file: `text_multiagent_finetuning.py`
- description: implement a multiagent finetuning loop where several copies of the same base model act as generation and critic agents. On the dev set, agents first run a multi-round debate: round 0 produces initial answers, later rounds let each agent see a summary of the other agents’ previous answers plus its own history and refine its response. For each iteration, we (1) majority-vote over the final extracted answers to get a consensus per question, (2) build per-agent generation datasets from examples where that agent’s *initial* answer agrees with the consensus, and (3) build per-agent critic datasets from full debate histories where the *final* answer agrees with the consensus, mixing “fixed my mistake” and “stayed correct” cases. Each generation and critic agent is then SFT-trained (LoRA) on its own dataset, and the process can repeat for multiple iterations. At test time, the finetuned agents run the same debate protocol and the final prediction is decided by majority vote over extracted answers.
- related paper(s):
    - [Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains](https://arxiv.org/abs/2501.05707)
- method-specific hyperparameters:
    - `iterations`, default 1: number of finetuning iterations (debate → build datasets → SFT).
    - `rounds`, default 2: total debate rounds (0 = initial generation, later rounds = critics).
    - `w`, default 0.5: relative weight of critic examples where the agent corrected an initially wrong answer versus examples where it stayed correct.
    - `training_ratio`, default 1.0: fraction of the dev split used to build the SFT datasets.
    - `sft_epochs`, default 1: number of epochs for LoRA SFT for each agent.
    - `sft_learning_rate`, default 1e-5: learning rate for SFT.
    - `sft_batch_size`, default 1: per-device batch size for SFT.
    - `sft_grad_accum`, default 16: gradient accumulation steps for SFT.
- notes:
    - Currently supports `task_type` in `["multiple_choice", "exact_match", "f1_match"]`, using the existing `data/eval.py` extraction logic for voting and scoring.
    - Use moderate-sized models and small `iterations` / `rounds` if compute is limited; each iteration runs a full multi-model debate plus SFT.

#### Text-level: SPARTA
- file: `text_sparta.py`
- description: implement SPARTA alignment algorithm, an iterative competition-based training approach. In each iteration: (1) models compete pairwise on instructions, with opponents selected based on reputation scores or randomly; (2) other models (judges) dynamically score the competition responses (judges are models that didn't participate in each specific pair); (3) model ratings are updated based on judge scores using an Elo-like rating system; (4) preference pairs are generated from competitions; (5) DPO training is applied to improve models using the preference pairs. After all iterations, all adapters from all iterations are evaluated on the dev set, and the best one is selected for test evaluation.
- related paper(s):
    - [SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat](https://arxiv.org/abs/2506.04721)
- method-specific hyperparameters:
    - `num_iterations`, default 3: number of Sparta iterations to run. Each iteration includes competition, judging, rating updates, and DPO training.
    - `current_iteration`, default 0: starting iteration number (for resuming from a previous run).
    - `num_instructions`, default 500: number of instructions to use for competition in each iteration.
    - `random_match_prob`, default 0.2: probability of random opponent selection (vs. reputation-based selection). With probability `random_match_prob`, a random opponent is chosen; otherwise, the opponent is selected from the top-K models with similar reputation scores.
    - `num_opponents`, default 3: number of top-K opponents to consider for matching based on reputation scores when not using random selection.
    - `judge_batch_size`, default 8: batch size for judge model generation.
    - `judge_rounds`, default 1: number of judging rounds per pair. Each round produces independent scores that are averaged.
    - `initial_k`, default 10.0: initial K value for rating updates. The K value controls how much model ratings can change after each competition (similar to the K-factor in Elo rating systems). Higher K means ratings change more quickly.
    - `min_k`, default 5.0: minimum K value (after decay). The K value decays over time to make rating updates more stable as training progresses.
    - `window_size`, default 10: window size for deviation calculation. Used to track the variance in rating changes over recent updates.
    - `min_deviation`, default 0.1: minimum deviation value for rating uncertainty estimation.
    - `epsilon`, default 0.01: small epsilon for numerical stability in rating calculations.
    - `decay_rate`, default 0.9: decay rate for K value. The K value is multiplied by `decay_rate` every `decay_steps` updates.
    - `decay_steps`, default 10: steps for K decay. After this many rating updates, the K value is decayed.
    - `scaling_factor`, default 20.0: scaling factor for rating updates. Divides the raw rating change to control the magnitude of updates.
    - `score_type`, default "normal": rating system type. Options: "normal" (standard Elo-like rating with equal weights for all judges), "dynamic" (dynamic-weighted, where model weights are computed based on previous iterations' performance, with lower-performing models getting reduced weights), or "static" (static-weighted, where weights are gradually assigned to more models over iterations based on their historical performance).
    - `freeze_ratings`, default false: if true, model ratings are not updated during the competition phase. Useful for testing or when you want to use fixed ratings.
    - `debug`, default false: if true, prints detailed rating update information (update count, K value, deviation changes) for debugging purposes.
#### Text-level: AggLM
- file: `agglm.py`
- description: trains an aggregator model to synthesize final solutions from multiple candidate solutions using reinforcement learning from verifiable rewards (RLVR). Given a problem and m candidate solutions from one or more LLMs, AggLM learns to review, reconcile, and combine them into a superior final answer. The method uses GRPO (Group-Relative Policy Optimization) with LoRA fine-tuning and carefully balances training on "hard" examples (where majority voting fails) and "easy" examples (where majority voting succeeds) to learn both minority-answer recovery and reliable aggregation.
- related paper(s):
    - [The Majority is not always right: RL training for solution aggregation](https://arxiv.org/pdf/2509.06870)
- method-specific hyperparameters:
    - `agg_model`, default `Qwen/Qwen3-1.7B`: the base model to initialize the aggregator. Should ideally have the largest vocabulary size among the solution models. Can be a HuggingFace Hub ID or local path.
    - `agglm_log_path`, default `logs/agglm`: directory to save training checkpoints, intermediate generation results, and trained LoRA adapters.
    - `reuse_log`, default `True`: whether to reuse existing generation and model weights. You might need to set this hyperparameter to `False` if error happens in the generation or training process
    - **Training hyperparameters:**
        - `learning_rate`, default 1e-4: learning rate for GRPO optimization.
        - `weight_decay`, default 1e-5: weight decay for regularization.
        - `lr_scheduler`, default `cosine`: learning rate scheduler type (e.g., `cosine`, `linear`).
        - `max_epoches`, default 10: number of training epochs.
        - `train_batch_size`, default 4: batch size for both GRPO training (per_device_train_batch_size and num_generations).
        - `sample_size`, default 2: number of solution sets (each containing m solutions) to sample per training problem for diversity. Increasing this introduces more variety in answer combinations but increases training data size linearly.
        - `max_response_length`, default 512. If you are using a thinking model, have a large max_response_length of at least 1024.
    - **LoRA configuration (fixed in code):**
      - rank: 64
      - lora_alpha: 16
      - lora_dropout: 0.1
      - target_modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
    - **GRPO-specific settings (fixed in code):**
        - group size, default 4: number of generations per GRPO update, same to `train_batch_size`
        - max_prompt_length: 4096
        - warmup_ratio: 0.1
- workflow:
    1. **Training data generation**: For each problem in the dev set, sample `sample_size * m` solutions from the `model_names` solution models (m = number of models). Group them into `simple_size` sets of m solutions each.
    2. **Hard/easy classification**: Evaluate each solution set. A set is "hard" if the majority answer (most frequent) is incorrect; otherwise it's "easy".
    3. **Balanced mixing**: Keep all hard examples and randomly sample an equal number of easy examples (50% by default) to create the final training dataset. This prevents overfitting to either always trusting majority vote or always ignoring it.
    4. **RL training**: Train the aggregator using GRPO with binary rewards (1 if aggregated answer matches ground truth, 0 otherwise) and the aggregation prompt template.
    5. **Inference**: Generate solutions from all solution models on the test set, aggregate them using the trained AggLM, and evaluate performance.
- warning: 
    - Training requires caching all dev set generations (can be large). Cached results are saved in `agglm_log_path` and reused across runs.
- note to tester: 
    - Recommended `model_names` for quick testing: `["Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B"]` (single model, can aggregate solutions from itself).
    - Outputs: 
      - Trained LoRA adapter: `{agglm_log_path}/{model_filename}/adapter_model.safetensors`
      - Cached generations: `{agglm_log_path}/{model_filename}.json`
      - Final results: `logs/{task}_{num_models}_{score}_agglm.json`

#### Logit-level: Logit Fusion
- file: `logit_logit_fusion.py`
- description: fuse the output logits of multiple LLMs and decode from the joint distribution. **All LLMs must share the same architecture and vocabulary.**
- method-specific hyperparameters:
    None
- warning: you might need very small batch sizes. len(gpu_ids) has to == len(model_names).
- note to tester: one recommended `model_names`: ["allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B"]. Try another set of your own choices.

#### Logit-level: Logit Contrastive
- file: `logit_logit_contrastive.py`
- description: contrast the logits of best- and worst-performing LLMs and decode from the contrastive distribution. **All LLMs must share the same architecture and vocabulary.** First, evaluate all LLMs on the dev set to decide the top-k and bottom-k. Then decode with P1 + ... + Pk - (Pk+1 + ... + Pn), where Pi is the output distribution of the i-th ranked LLM.
- related paper(s):
    - [DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts](https://arxiv.org/abs/2105.03023)
    - [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565)
- method-specific hyperparameters:
    - `k`, default 1: the number of top and bottom LLMs to use for contrastive decoding.
- warning: you might need very small batch sizes. len(gpu_ids) has to == len(model_names).
- note to tester: one recommended `model_names`: ["allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B"]. Try another set of your own choices.

#### Weight-level: Greedy Soup
- file: `weight_greedy_soup.py`
- description: average the weights of multiple LLMs in a greedy manner. **All LLMs must share the same architecture.** First, evaluate all LLMs on the dev set and sort them by performance. Then, starting from the best model, iteratively add one model at a time to the soup if it improves performance on the dev set. We provide a bridge to the MergeKit implmementation.
- related paper(s):
    - [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)
- method-specific hyperparameters:
    None
- note to tester: one recommended `model_names`: ["allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B"]. Try another set of your own choices.

#### Weight-level: Dare Ties
- file: `weight_dare_ties.py`
- description: average the weights of multiple LLMs with DARE-TIES. **All LLMs must share the same architecture.** Two modes: the average mode, where models have equal weight, the optimized mode, where weights are optimized on the dev set by particle swarm optimization. We provide a bridge to the MergeKit implmementation.
- related paper(s):
    - [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)
    - [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- method-specific hyperparameters:
    - `base_model_name`: the common base that these finetuned models share. For example, `Qwen/Qwen2.5-7B-Instruct` for ["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt", "bunsenfeng/yuru_qw_oasst1"].
    - `mode`, default `average`: `average` or `optimized`.
    - `population`, default 5: the population size for particle swarm optimization (only used in `optimized` mode).
    - `max_iterations`, default 5: the maximum number of iterations for particle swarm optimization (only used in `optimized` mode).
    - There are more hyperparameters in `optimized` mode for particle swarm optimization, please refer to `weight_dare_ties.py`. Only change them if you know what you are doing.
- note to tester: one recommended `model_names`: base_model_name `Qwen/Qwen2.5-7B-Instruct` for ["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt", "bunsenfeng/yuru_qw_oasst1"]. Try another set of your own choices.

#### Weight-level: Model Swarms
- file: `weight_model_swarms.py`
- description: multiple LMs collaboratively search in the weight space to find better model weights. **All LLMs must share the same architecture.**
- related paper(s):
    - [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)
- method-specific hyperparameters:
    - `swarm_base_path`, default `logs/model_swarms/`: a place to do the bookkeeping for the swarm algorithm.
    - `base_model`, default None: the common base model architecture of these LLMs.
    - `fast_merge_flag`, default False: False if they are regular full-size models, True if they are lora adapters.
    - `max_iterations`, default 10: the maximum number of iterations.
    - There are more hyperparameters for particle swarm optimization, please refer to `weight_model_swarms.py`. Only change them if you know what you are doing.
- warning: HIGHLY recommended to use lora adapters and set `fast_merge_flag` to True to save computation and memory.
- note to tester: recommended set of `model_names`: ["bunsenfeng/ds_science", "bunsenfeng/ds_oasst1", "bunsenfeng/ds_lima"] and ["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt", "bunsenfeng/yuru_qw_oasst1"]. These two settings could use `fast_merge_flag` as True which is wayyyyy faster. Optionally, try a set of full-sized models (not lora adapters) with `fast_merge_flag` as False if you have enough computation resources.

#### Weight-level: PhatGoose
- file: `weight_phatgoose.py`
- description: full PhatGoose-style mixture-of-experts with per-token × per-module top-k routing over multiple LoRA experts. **All experts must be LoRA adapters of the same base model.** At each LoRA injection site, each expert has a learnable gate vector that scores token activations via normalized dot product. During inference, top-k experts are selected per token per module, and their LoRA deltas are combined with softmax-weighted routing. Gate vectors are trained separately per expert (with base model and LoRA weights frozen) using SFT loss on the task's dev set data.
- related paper(s):
    - [Learning to Route Among Specialized Experts for Zero-Shot Generalization](https://arxiv.org/abs/2402.05859)
- method-specific hyperparameters:
    - `mode`, default `train_and_infer` for end-to-end training and evaluation. Available modes:
        - `train_and_infer`: train gate vectors for all experts, then immediately run inference.
        - `train_all_gates`: train gate vectors for all experts only (no inference). Use if you want to save gates for later reuse.
        - `infer_moe_full`: inference only (requires pre-trained gates). Use if you already have trained gate checkpoints.
    - `base_model`, default auto-detect: the base model that all LoRA experts share. **The code will automatically detect this from the first adapter's config file.** You can also manually specify a HuggingFace Hub ID or local path if needed.
    - `tokenizer_name`, default `base_model`: tokenizer name or path.
    - **Gate training hyperparameters (for `train_and_infer` and `train_all_gates` modes):**
        - `gate_steps`, default 100: number of training steps per expert for gate learning.
        - `gate_batch_size`, default 1: batch size for gate training.
        - `gate_lr`, default 0.005: learning rate for gate training (AdamW optimizer). Matches the original PhatGoose implementation.
        - `max_length`, default 512: maximum sequence length for gate training.
        - `grad_accum`, default 1: gradient accumulation steps during gate training. Original paper uses 32, but 1 works with small batch sizes.
        - `gate_output_dir`, default `model_collaboration/logs/phatgoose/<timestamp>/gates`: directory to save trained gate checkpoints.
    - **Inference hyperparameters (for `train_and_infer` and `infer_moe_full` modes):**
        - `gate_paths`, no default (required for `infer_moe_full` mode only): list of paths to trained gate checkpoint files (`.pt`), one per expert. Not needed for `train_and_infer` as gates are auto-loaded after training.
        - `top_k`, default 2: number of experts to activate per token per module during inference.
        - `score_type`, default `cosine`: scoring function for routing. Options: `cosine` (normalized dot product) or `dot` (unnormalized).
        - `max_response_length`, default 128: maximum number of tokens to generate.
        - `temperature`, default 0.7: sampling temperature.
        - `top_p`, default 0.9: nucleus sampling parameter.
        - `batch_size`, default 8: batch size for inference generation.
        - `output_log_path`, default `model_collaboration/logs/<task>_<num_experts>_<score>_phatgoose.json`: path to save inference results.
- warning: **All LoRA experts must be adapters of the same base model.** LoRA target modules must be `nn.Linear` layers. Currently only supports causal language models (AutoModelForCausalLM). len(gpu_ids) can be 1; the method does not require multi-GPU.
- note to tester: recommended `model_names`: `["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt", "bunsenfeng/yuru_qw_oasst1"]` (LoRA adapters of Qwen2.5-7B-Instruct). These will be automatically downloaded from HuggingFace Hub. Use `mode: train_and_infer` for simplest testing.

#### Weight-level: LoraHub
- file: `weight_lorahub.py`
- description: Uses gradient-free optimization (Nevergrad) to learn the best scalar weights to linearly compose multiple LoRA adapters. The optimization minimizes the error (maximizes accuracy) on a few-shot development set using direct generation. **All LoRA adapters must share the same base model architecture.**
- related paper(s):
    - [LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269)
- method-specific hyperparameters:
    - `lorahub_dev_samples`, default 5: the number of few-shot examples from the dev set used to calculate the score during the optimization loop.
    - `max_inference_step`, default 20: the maximum number of iterations (budget) for the optimizer. Since generation is performed at every step, keep this number reasonable.
    - `lora_weight_bound`, default 1.5: the boundary for the search space of the adapter weights (e.g., search within [-1.5, 1.5]).
    - `regular_coef`, default 0.05: the coefficient for L1 regularization to prevent weights from becoming too large.
- warning: **All LoRA adapters must share the same base model architecture.** This implementation optimizes based on **generation accuracy**, not likelihood loss. This ensures the metric aligns with the final goal.
- note to tester: recommended set of `model_names`: ["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt", "bunsenfeng/yuru_qw_oasst1"].

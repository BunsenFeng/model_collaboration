"""
Logit-level: Co-LLM (Collaborative Language Model Decoding)

Inspired by "Learning to Decode Collaboratively with Multiple Language Models" (ACL 2024)
Paper: https://arxiv.org/pdf/2403.038704.16792

This method uses the deferral model to score the data and the training model to train the model.
The deferral model is used to score the data and the training model is used to train the model.
Collaborative inference is used to generate the final response.

Workflow:
1. Initialize training data
2. Score data with generator model
3. Score data with mentor model
4. Initialize training data with deferral model
5. Train model with deferral model
6. Generate final response with collaborative inference
"""

import os
import json
import torch
import logging
import datasets
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import TrainingArguments
from model_collaboration.data import eval
from model_collaboration.utils.collm_scoring import CoLLMScorer
from model_collaboration.utils.collm_training import CoLLMTrainer

SYSTEM_PROMPT = "You are a helpful assistant that solves problems step by step."

logger = logging.getLogger(__name__)

class TrainDataInitializer:
    """
    Initialize training data from a dataset.

    Args:
        dataset_name (str): Name of the dataset to load (default: 'nlile/hendrycks-MATH-benchmark')
        split (str): Dataset split to use (default: 'train')
        training_num (int): Number of training examples to use (default: 10000)
        output_dir (str): Output directory for processed data
        output_name (str): Output filename (default: 'math_data.jsonl')
    """
    def __init__(
        self,
        dataset_name='nlile/hendrycks-MATH-benchmark',
        split='train',
        training_num=10000,
        output_dir='data/train_data_initializer',
        output_name='math_data.jsonl'
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.training_num = training_num
        self.output_dir = output_dir
        self.output_name = output_name
        self.output_path = os.path.join(output_dir, output_name)

        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded with {len(self.dataset)} examples")

    def process(self):
        """Process the dataset and save to JSONL format."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Determine number of examples to process
        num_examples = min(self.training_num, len(self.dataset))
        logger.info(f"Processing {num_examples} training examples...")

        with open(self.output_path, 'w') as f:
            for idx, example in enumerate(tqdm(self.dataset.select(range(num_examples)), desc="Processing training data")):
                problem = example["problem"]
                solution = example["solution"]
                level = example.get("level", "unknown")
                type_name = example.get("subject", "unknown")

                # Create messages without few-shot examples
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Problem: {problem.strip()}"
                    },
                    {
                        "role": "assistant",
                        "content": solution.strip()
                    }
                ]

                f.write(
                    json.dumps({
                        "dataset": self.dataset_name.split('/')[-1],
                        "id": f"{self.split}_{idx}",
                        "level": level,
                        "type": type_name,
                        "messages": messages,
                    }, ensure_ascii=False) + "\n"
                )

        logger.info(f"Training data saved to: {self.output_path}")


class DataInitializer:
    def __init__(self, training_model_ds="default", deferral_model_ds="default", init_method="default", output_dir="default", log_level="DEBUG"):
        self.training_model_ds = training_model_ds
        self.deferral_model_ds = deferral_model_ds
        self.init_method = init_method
        self.output_dir = output_dir
        self.log_level = log_level
        logger.debug("Base  df  (Small Model)  path : {}".format(self.training_model_ds))
        logger.debug("Deferral df (Large Model) path: {}".format(self.deferral_model_ds))
        logger.debug("Output path:                    {}".format(self.output_dir))
        logger.debug("Init method: {}".format(self.init_method))
        self.ds1 = datasets.load_from_disk(self.deferral_model_ds)
        self.ds2 = datasets.load_from_disk(self.training_model_ds)
        assert len(self.ds1) == len(self.ds2)

    def get_data(self):
        all_more_confidence = []
        for example_id in range(len(self.ds1)):
            is_reference_max_deferral = torch.isclose(
                self.ds1[example_id]["reference_log_probs"],
                self.ds1[example_id]["reference_max_prob"],
            )
            is_reference_max_training = torch.isclose(
                self.ds2[example_id]["reference_log_probs"],
                self.ds2[example_id]["reference_max_prob"],
            )

            if self.init_method == "a":
                # When the deferral model is correct, while the training model is not
                more_confident = is_reference_max_deferral > is_reference_max_training
                more_confident = more_confident.long()
                ## Some additional note for why this is correct
                # more_confident = torch.logical_and(
                #     torch.logical_or(is_reference_max_deferral, is_reference_max_training), ~is_reference_max_training
                # )
                # more_confident = more_confident.long()
                # a = torch.logical_and(
                #     torch.logical_or(is_reference_max_deferral, is_reference_max_training), ~is_reference_max_training
                # )
                # b = is_reference_max_deferral > is_reference_max_training
                # assert torch.allclose(a, b)

            elif self.init_method == "b":
                # When the training model is not correct.
                more_confident = (~is_reference_max_training).long()
            else:
                raise NotImplementedError("Unknown init method {}".format(self.init_method))

            # We want to mask out the log probs in user conversations, similar to what
            # the tulu paper does
            input_mask = self.ds1[example_id]["labels"][1:] == -100
            more_confident[input_mask] = -100
            all_more_confidence.append(more_confident)

        ds3 = self.ds1.remove_columns(["reference_log_probs"]).add_column(
            "reference_log_probs", [ele.long().tolist() for ele in all_more_confidence]
        )
        ds3.save_to_disk(self.output_dir)

        no_zero = (torch.cat(all_more_confidence) == 1).float().mean()
        logger.debug("Average non-zero ratio: {}".format(no_zero))


class ModelScorer:
    """
    A wrapper class for scoring datasets with models using CoLLMScorer.

    This class provides a convenient interface to score datasets, similar to the
    command-line usage of the scoring script.

    Example:
        scorer = ModelScorer(
            model_name_or_path="/path/to/model",
            tokenizer_name="/path/to/tokenizer",
            train_file="data/processed/dataset/data.jsonl",
            output_dir="checkpoints/dataset/model_name",
            max_seq_length=2048,
            use_flash_attn=True,
            use_completion_format=True,
            use_slow_tokenizer=True,
            preprocessing_num_workers=16
        )
        scorer.run()
    """

    def __init__(
        self,
        model_name_or_path,
        tokenizer_name,
        train_file,
        output_dir,
        max_seq_length=512,
        use_flash_attn=False,
        use_completion_format=False,
        use_slow_tokenizer=False,
        preprocessing_num_workers=None,
        preprocessing_format="tulu_chat",
        dataset_name=None,
        dataset_config_name=None,
        max_train_samples=None,
        overwrite_cache=False,
        debug=False,
        local_files_only=False,
    ):
        """
        Initialize the ModelScorer.

        Args:
            model_name_or_path: Path to pretrained model or model identifier
            tokenizer_name: Pretrained tokenizer name or path
            train_file: Path to training file (json/jsonl)
            output_dir: Directory to save the scored dataset
            max_seq_length: Maximum sequence length for tokenization (default: 512)
            use_flash_attn: Whether to use flash attention (default: False)
            use_completion_format: Whether to use completion format (default: False)
            use_slow_tokenizer: Whether to use slow tokenizer (default: False)
            preprocessing_num_workers: Number of workers for preprocessing (default: None)
            preprocessing_format: Format for preprocessing, "tulu_chat" or "llama2_chat" (default: "tulu_chat")
            dataset_name: Name of the dataset to load (default: None)
            dataset_config_name: Configuration name of the dataset (default: None)
            max_train_samples: Maximum number of training samples (default: None)
            overwrite_cache: Whether to overwrite cache (default: False)
            debug: Whether to run in debug mode (default: False)
            local_files_only: Whether to only use local cached files (default: False)
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name = tokenizer_name
        self.train_file = train_file
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.use_flash_attn = use_flash_attn
        self.use_completion_format = use_completion_format
        self.use_slow_tokenizer = use_slow_tokenizer
        self.preprocessing_num_workers = preprocessing_num_workers
        self.preprocessing_format = preprocessing_format
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.max_train_samples = max_train_samples
        self.overwrite_cache = overwrite_cache
        self.debug = debug
        self.local_files_only = local_files_only

        # Initialize the scorer
        self.scorer = None

    def run(self):
        """
        Run the scoring process.

        This method initializes the CoLLMScorer and processes the dataset.

        Returns:
            Scored dataset
        """
        logger.info(f"Initializing ModelScorer for model: {self.model_name_or_path}")
        logger.info(f"Train file: {self.train_file}")
        logger.info(f"Output dir: {self.output_dir}")

        # Initialize CoLLMScorer
        self.scorer = CoLLMScorer(
            model_name_or_path=self.model_name_or_path,
            tokenizer_name=self.tokenizer_name,
            max_seq_length=self.max_seq_length,
            preprocessing_format=self.preprocessing_format,
            use_slow_tokenizer=self.use_slow_tokenizer,
            use_flash_attn=self.use_flash_attn,
            local_files_only=self.local_files_only,
        )

        # Process the dataset
        scored_data = self.scorer.process(
            dataset_name=self.dataset_name,
            dataset_config_name=self.dataset_config_name,
            train_file=self.train_file,
            output_dir=self.output_dir,
            max_train_samples=self.max_train_samples,
            use_completion_format=self.use_completion_format,
            preprocessing_num_workers=self.preprocessing_num_workers,
            overwrite_cache=self.overwrite_cache,
            debug=self.debug,
        )

        logger.info(f"Scoring completed. Results saved to: {self.output_dir}")
        return scored_data


class ModelTrainer:
    """
    A wrapper class for training models with deferral mechanism.

    This class provides a convenient interface to train models using CoLLMTrainer,
    similar to command-line usage.

    Example:
        trainer = ModelTrainer(
            model_name_or_path="/path/to/model",
            output_dir="checkpoints/output",
            train_file="data/train.jsonl",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            use_flash_attn=True
        )
        metrics = trainer.run()
    """

    def __init__(
        self,
        model_name_or_path,
        output_dir,
        train_file=None,
        dataset_name=None,
        dataset_config_name=None,
        # Model arguments
        tokenizer_name=None,
        config_name=None,
        use_flash_attn=False,
        torch_dtype=None,
        cache_dir=None,
        use_fast_tokenizer=False,
        # Data arguments
        max_seq_length=None,
        max_train_samples=None,
        preprocessing_num_workers=None,
        streaming=False,
        overwrite_cache=False,
        # Deferral arguments
        no_deferral_initialization_search=False,
        deferral_initialization_search_max_steps=4000,
        deferral_initialization_weight_balance=None,
        deferral_trainer_version="v1",
        deferral_initialization_search_version="v1",
        deferral_initialization_path=None,
        # Training arguments
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        deepspeed=None,
        local_rank=-1,
        seed=42,
        dataloader_num_workers=0,
        **training_kwargs,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model_name_or_path: Path to pretrained model
            output_dir: Directory to save the trained model
            train_file: Path to training file
            dataset_name: Name of the dataset to use
            dataset_config_name: Configuration name of the dataset
            tokenizer_name: Pretrained tokenizer name or path
            config_name: Pretrained config name or path
            use_flash_attn: Whether to use flash attention
            torch_dtype: Torch dtype for model
            cache_dir: Cache directory
            use_fast_tokenizer: Whether to use fast tokenizer
            max_seq_length: Maximum sequence length
            max_train_samples: Maximum number of training samples
            preprocessing_num_workers: Number of preprocessing workers
            streaming: Enable streaming mode
            overwrite_cache: Overwrite cached datasets
            no_deferral_initialization_search: Skip deferral initialization search
            deferral_initialization_search_max_steps: Max steps for initialization search
            deferral_initialization_weight_balance: Weight balance for deferral token
            deferral_trainer_version: Version of deferral trainer
            deferral_initialization_search_version: Version of initialization search
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device for training
            per_device_eval_batch_size: Batch size per device for evaluation
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            lr_scheduler_type: Learning rate scheduler type
            logging_steps: Logging steps
            save_steps: Save checkpoint every N steps
            save_total_limit: Limit the total number of checkpoints
            fp16: Use fp16 training
            bf16: Use bf16 training
            gradient_checkpointing: Use gradient checkpointing
            deepspeed: DeepSpeed config file path
            local_rank: Local rank for distributed training
            seed: Random seed
            dataloader_num_workers: Number of dataloader workers
            **training_kwargs: Additional training arguments
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir

        # Create TrainingArguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            deepspeed=deepspeed,
            local_rank=local_rank,
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
            do_train=True,
            **training_kwargs,
        )

        # Create CoLLMTrainer
        self.trainer = CoLLMTrainer(
            model_name_or_path=model_name_or_path,
            output_dir=output_dir,
            train_file=train_file,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            tokenizer_name=tokenizer_name,
            config_name=config_name,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            preprocessing_num_workers=preprocessing_num_workers,
            cache_dir=cache_dir,
            use_fast_tokenizer=use_fast_tokenizer,
            use_flash_attn=use_flash_attn,
            torch_dtype=torch_dtype,
            no_deferral_initialization_search=no_deferral_initialization_search,
            deferral_initialization_search_max_steps=deferral_initialization_search_max_steps,
            deferral_initialization_weight_balance=deferral_initialization_weight_balance,
            deferral_trainer_version=deferral_trainer_version,
            deferral_initialization_search_version=deferral_initialization_search_version,
            deferral_initialization_path=deferral_initialization_path,
            streaming=streaming,
            overwrite_cache=overwrite_cache,
            training_args=self.training_args,
        )

    def run(self, resume_from_checkpoint=None):
        """
        Run the complete training pipeline.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        logger.info(f"Starting training for model: {self.model_name_or_path}")
        logger.info(f"Output directory: {self.output_dir}")

        metrics = self.trainer.run(
            training_args=self.training_args, resume_from_checkpoint=resume_from_checkpoint
        )

        logger.info("Training completed successfully!")
        return metrics


class CoLLMInference:
    """
    A class for running inference with trained Co-LLM models.

    Uses the trained deferral model (generator) and mentor model for collaborative inference.
    """

    def __init__(
        self,
        model_base,
        model_ref,
        device_base="cuda:0",
        device_ref="cuda:1",
        max_tokens=2048,
        deferral_threshold=0.5,
        deferral_strategy="defer",
        threshold_warmup_schedule="none",
        threshold_warmup_steps=15,
    ):
        """
        Initialize the CoLLMInference.

        Args:
            model_base: Path to base model (trained deferral model)
            model_ref: Path to reference model (mentor/large model)
            device_base: Device for base model
            device_ref: Device for reference model
            max_tokens: Maximum tokens to generate
            deferral_threshold: Threshold for deferring to reference model
            deferral_strategy: Strategy for deferral ("defer" or "compose")
            threshold_warmup_schedule: Warmup schedule for threshold
            threshold_warmup_steps: Number of warmup steps
        """
        self.model_base_path = model_base
        self.model_ref_path = model_ref
        self.device_base = device_base
        self.device_ref = device_ref
        self.max_tokens = max_tokens
        self.deferral_threshold = deferral_threshold
        self.deferral_strategy = deferral_strategy
        self.threshold_warmup_schedule = threshold_warmup_schedule
        self.threshold_warmup_steps = threshold_warmup_steps

        # Load models
        self._load_models()

    def _load_models(self):
        """Load both base and reference models."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading base model from {self.model_base_path}...")
        self.tokenizer_base = AutoTokenizer.from_pretrained(
            self.model_base_path, trust_remote_code=True
        )

        device_map_base = "auto" if "," in self.device_base else self.device_base
        self.model_base = AutoModelForCausalLM.from_pretrained(
            self.model_base_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map_base,
            trust_remote_code=True,
        )
        self.model_base.eval()

        # Check for deferral token
        self.has_defer_token = "<|defer|>" in self.tokenizer_base.get_vocab()
        if self.has_defer_token:
            self.defer_token_id = self.tokenizer_base.convert_tokens_to_ids("<|defer|>")
            logger.info(f"Found <|defer|> token at ID: {self.defer_token_id}")
        else:
            self.defer_token_id = len(self.tokenizer_base) - 1
            logger.warning("No <|defer|> token found, using last token ID")

        logger.info(f"Loading reference model from {self.model_ref_path}...")
        self.tokenizer_ref = AutoTokenizer.from_pretrained(
            self.model_ref_path, trust_remote_code=True
        )

        device_map_ref = "auto" if "," in self.device_ref else self.device_ref
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_ref_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map_ref,
            trust_remote_code=True,
        )
        self.model_ref.eval()

        logger.info("Models loaded successfully")

    def _create_deferral_threshold_schedule(self, total_steps):
        """Create a schedule for deferral thresholds with warmup."""
        import numpy as np

        if self.threshold_warmup_schedule is None or self.threshold_warmup_schedule == "none":
            return torch.tensor([self.deferral_threshold] * total_steps)

        warmup_steps = min(self.threshold_warmup_steps, total_steps)
        start_value = 1.0
        end_value = self.deferral_threshold

        if self.threshold_warmup_schedule == "linear":
            warmup_values = torch.linspace(start_value, end_value, warmup_steps)
        elif self.threshold_warmup_schedule == "cosine":
            warmup_values = torch.tensor([
                end_value + (start_value - end_value) * 0.5 * (1 + np.cos(np.pi * i / warmup_steps))
                for i in range(warmup_steps)
            ])
        else:
            raise ValueError(f"Unknown warmup schedule: {self.threshold_warmup_schedule}")

        remaining_values = torch.tensor([end_value] * (total_steps - warmup_steps))
        return torch.cat([warmup_values, remaining_values])

    def generate(self, prompt, verbose=False):
        """
        Generate response with deferral mechanism.

        Args:
            prompt: Input prompt (str)
            verbose: Whether to print verbose output

        Returns:
            Generated text (str)
        """
        # Tokenize prompt
        prompt_token_ids = self.tokenizer_base(prompt, return_tensors="pt")["input_ids"][0].tolist()

        vocab_size_base = len(self.tokenizer_base)
        vocab_size_ref = len(self.tokenizer_ref)
        eos_token_id = self.tokenizer_base.eos_token_id

        # Create deferral threshold schedule
        deferral_thresholds = self._create_deferral_threshold_schedule(self.max_tokens)

        total_generated = 0
        generated = []
        eos_reached = False

        # Convert to tensor
        input_ids = torch.tensor([prompt_token_ids], device=self.model_base.device)

        with torch.no_grad():
            while not eos_reached and total_generated < self.max_tokens:
                # Get logits from base model
                outputs_base = self.model_base(input_ids)
                logits_base = outputs_base.logits[:, -1, :]

                # Extract deferral probability
                deferral_logit = logits_base[:, self.defer_token_id]
                deferral_prob = torch.sigmoid(deferral_logit).item()

                # Mask out deferral token
                logits_base_masked = logits_base.clone()
                logits_base_masked[:, self.defer_token_id] = float('-inf')

                # Get next token from base model
                next_token_base = logits_base_masked.argmax(dim=-1).item()

                # Check if we should defer
                cur_threshold = deferral_thresholds[total_generated].item()
                should_defer = deferral_prob > cur_threshold

                if should_defer:
                    if verbose:
                        logger.info(f"Token {total_generated}: Deferring (prob={deferral_prob:.3f} > thresh={cur_threshold:.3f})")

                    # Get prediction from reference model
                    input_ids_ref = input_ids.to(self.model_ref.device)
                    outputs_ref = self.model_ref(input_ids_ref)
                    logits_ref = outputs_ref.logits[:, -1, :]

                    if self.deferral_strategy == "defer":
                        # Use reference model's token directly
                        next_token = logits_ref.argmax(dim=-1).item()
                    elif self.deferral_strategy == "compose":
                        # Compose probabilities
                        probs_base = torch.softmax(logits_base_masked[:, :vocab_size_ref], dim=-1)
                        probs_ref = torch.softmax(logits_ref[:, :vocab_size_ref], dim=-1)
                        probs_ref = probs_ref.to(probs_base.device)
                        combined_probs = (1 - deferral_prob) * probs_base + deferral_prob * probs_ref
                        next_token = combined_probs.argmax(dim=-1).item()
                    else:
                        raise ValueError(f"Unknown deferral strategy: {self.deferral_strategy}")
                else:
                    # Use base model's token
                    next_token = next_token_base

                # Append token
                generated.append(next_token)
                total_generated += 1

                # Check for EOS
                if next_token == eos_token_id:
                    eos_reached = True
                    break

                # Check for stop patterns
                generated_text_so_far = self.tokenizer_base.decode(generated, skip_special_tokens=True)
                if "\n\nProblem" in generated_text_so_far or "\n\nQuestion" in generated_text_so_far:
                    eos_reached = True
                    break

                # Update input
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.model_base.device)], dim=1)

        # Decode generated text
        generated_text = self.tokenizer_base.decode(generated, skip_special_tokens=True)
        return generated_text

    def batch_generate(self, prompts, verbose=False):
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            verbose: Whether to print verbose output

        Returns:
            List of generated texts
        """
        outputs = []
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            if verbose and i == 0:
                output = self.generate(prompt, verbose=True)
            else:
                output = self.generate(prompt, verbose=False)
            outputs.append(output)
        return outputs

def run_inference(
    task,
    task_type,
    split="test",
    model_base=None,
    model_ref=None,
    gpu_ids=None,
    deferral_threshold=0.5,
    deferral_strategy="defer",
    max_tokens=2048,
    threshold_warmup_schedule="none",
    threshold_warmup_steps=15,
    save_results=True,
):
    """
    Run inference with trained Co-LLM models.

    Args:
        task: Task name (e.g., 'math')
        task_type: Type of task (e.g., 'exact_match', 'f1_match')
        split: Dataset split ('test', 'dev')
        model_base: Path to trained base model (deferral model)
        model_ref: Path to reference model (mentor)
        gpu_ids: List of GPU IDs [base_gpu, ref_gpu]
        deferral_threshold: Threshold for deferring
        deferral_strategy: Strategy ("defer" or "compose")
        max_tokens: Max tokens to generate
        threshold_warmup_schedule: Warmup schedule
        threshold_warmup_steps: Warmup steps
        save_results: Whether to save results

    Returns:
        Average score
    """
    import json
    from pathlib import Path

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # Set GPU devices
    if gpu_ids is None:
        gpu_ids = [0, 1]
    device_base = f"cuda:{gpu_ids[0]}"
    device_ref = f"cuda:{gpu_ids[1]}"

    logger.info("=" * 80)
    logger.info("Co-LLM Inference")
    logger.info("=" * 80)
    logger.info(f"Task: {task}")
    logger.info(f"Task type: {task_type}")
    logger.info(f"Split: {split}")
    logger.info(f"Base model: {model_base}")
    logger.info(f"Reference model: {model_ref}")
    logger.info(f"Deferral threshold: {deferral_threshold}")
    logger.info(f"Deferral strategy: {deferral_strategy}")
    logger.info("=" * 80)

    # Prepare inputs using eval.prepare_inputs
    logger.info("Preparing inputs...")
    test_input_list = eval.prepare_inputs(task, task_type, split)
    logger.info(f"Loaded {len(test_input_list)} inputs")

    # Initialize inference
    logger.info("Initializing Co-LLM inference...")
    inference = CoLLMInference(
        model_base=model_base,
        model_ref=model_ref,
        device_base=device_base,
        device_ref=device_ref,
        max_tokens=max_tokens,
        deferral_threshold=deferral_threshold,
        deferral_strategy=deferral_strategy,
        threshold_warmup_schedule=threshold_warmup_schedule,
        threshold_warmup_steps=threshold_warmup_steps,
    )

    # Generate outputs
    logger.info("Generating outputs...")
    outputs = inference.batch_generate(test_input_list, verbose=False)

    # Evaluate
    logger.info("Evaluating outputs...")
    test_scores = eval.get_scores(task, task_type, split, outputs)
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0

    logger.info("=" * 80)
    logger.info(f"Final {split} {task} score: {avg_test_score:.4f}")
    logger.info("=" * 80)

    # Save results
    if save_results:
        experiment_logs = {
            "task": task,
            "task_type": task_type,
            "split": split,
            "model_base": model_base,
            "model_ref": model_ref,
            "deferral_threshold": deferral_threshold,
            "deferral_strategy": deferral_strategy,
            "avg_score": avg_test_score,
            "logs": []
        }

        for i in range(len(test_input_list)):
            log = {
                "input": test_input_list[i],
                "output": outputs[i],
                "score": test_scores[i]
            }
            experiment_logs["logs"].append(log)

        # Create log filename
        base_name = model_base.split("/")[-1] if "/" in model_base else model_base
        ref_name = model_ref.split("/")[-1] if "/" in model_ref else model_ref
        log_filename = f"model_collaboration/logs/{task}_{base_name}_{ref_name}_collm_thresh{deferral_threshold:.2f}.json"

        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        with open(log_filename, "w") as f:
            json.dump(experiment_logs, f, indent=4)

        logger.info(f"Results saved to: {log_filename}")

    return avg_test_score


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Run the Co-LLM method for model collaboration.

    Args:
        task: Task name (e.g., 'math')
        task_type: Type of task
        gpu_ids: GPU IDs to use
        model_names: List of two model names [generator, mentor]
        hyperparameters: Dict of hyperparameters with the following keys:
            # Dataset parameters
            - training_dataset_name (str): Name of the training dataset (default: "nlile/hendrycks-MATH-benchmark")
            - training_split (str): Dataset split for training (default: "train")
            - training_num (int): Number of training examples to use (default: 10000)

            # Inference control
            - run_inference (bool): Whether to run inference after training (default: True)
            - inference_split (str): Dataset split for inference (default: "test")

            # Deferral parameters
            - deferral_threshold (float): Threshold for deferring to mentor (default: 0.5)
            - deferral_strategy (str): Strategy for deferral ("defer" or "compose") (default: "defer")
            - threshold_warmup_schedule (str): Warmup schedule ("none", "linear", "cosine") (default: "none")
            - threshold_warmup_steps (int): Number of warmup steps (default: 15)

            # Generation parameters
            - max_tokens (int): Maximum tokens to generate (default: 2048)

            # Output control
            - save_inference_results (bool): Whether to save inference results (default: True)

    Returns:
        str: Path to the Co-LLM directory containing checkpoints
    """
    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    if len(model_names) != 2:
        raise ValueError("Co-LLM requires exactly 2 models.")

    # Extract hyperparameters with defaults
    training_dataset_name = hyperparameters.get('training_dataset_name', 'nlile/hendrycks-MATH-benchmark')
    training_split = hyperparameters.get('training_split', 'train')
    training_num = hyperparameters.get('training_num', 10000)
    training_devices = hyperparameters.get('training_devices', [4, 5, 6, 7])

    # Convert training_devices to string
    devices_str = ','.join(map(str, training_devices))
    num_gpus = len(training_devices)

    generator = model_names[0]
    mentor = model_names[1]
    collm_name = f"{generator}-{mentor}"
    collm_dir = f"model_collaboration/logs/co-llm/{collm_name}"

    logger.info("=" * 80)
    logger.info(f"Training devices: GPU {devices_str} ({num_gpus} GPUs)")
    logger.info("=" * 80)

    # Set CUDA_VISIBLE_DEVICES for training
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    logger.info(f"Set CUDA_VISIBLE_DEVICES={devices_str}")

    if not os.path.exists(f"{collm_dir}/model_checkpoints_final"):
        os.makedirs(collm_dir, exist_ok=True)

        # Step 1: Initialize training data
        logger.info("Step 1: Initializing training data...")
        logger.info(f"  Dataset: {training_dataset_name}")
        logger.info(f"  Split: {training_split}")
        logger.info(f"  Number of examples: {training_num}")

        train_data = TrainDataInitializer(
            dataset_name=training_dataset_name,
            split=training_split,
            training_num=training_num,
            output_dir=collm_dir,
            output_name='math_data.jsonl'
        )
        train_data.process()

        # Step 2: Score data with generator model
        logger.info("Step 2: Scoring data with generator model...")
        generator_scorer = ModelScorer(
            model_name_or_path=generator,
            tokenizer_name=generator,
            train_file=f"{collm_dir}/math_data.jsonl",
            output_dir=f"{collm_dir}/generator_scored_data",
            max_seq_length=2048,
            use_flash_attn=True,
            use_completion_format=True,
            use_slow_tokenizer=True,
            preprocessing_num_workers=16
        )
        generator_scorer.run()

        # Step 3: Score data with mentor model
        logger.info("Step 3: Scoring data with mentor model...")
        mentor_scorer = ModelScorer(
            model_name_or_path=mentor,
            tokenizer_name=mentor,
            train_file=f"{collm_dir}/math_data.jsonl",
            output_dir=f"{collm_dir}/mentor_scored_data",
            max_seq_length=2048,
            use_flash_attn=True,
            use_completion_format=True,
            use_slow_tokenizer=True,
            preprocessing_num_workers=16
        )
        mentor_scorer.run()

        # Step 4: Initialize training data with deferral labels
        logger.info("Step 4: Initializing training data with deferral labels...")
        data_initializer = DataInitializer(
            training_model_ds=f"{collm_dir}/generator_scored_data",
            deferral_model_ds=f"{collm_dir}/mentor_scored_data",
            init_method="a",
            output_dir=f"{collm_dir}/train_data_initializer"
        )
        data_initializer.get_data()

        # ========== Phase 1: Deferral Token Initialization Search ==========
        logger.info("=" * 80)
        logger.info("Phase 1: Deferral token initialization search...")
        logger.info("=" * 80)

        phase1_trainer = ModelTrainer(
            model_name_or_path=generator,
            tokenizer_name=generator,
            output_dir=f"{collm_dir}/model_checkpoints_init",
            dataset_name=f"{collm_dir}/train_data_initializer",
            # Model configuration
            use_fast_tokenizer=False,
            use_flash_attn=True,
            torch_dtype="bfloat16",
            max_seq_length=2048,
            # Deferral configuration
            deferral_initialization_weight_balance=8,
            deferral_initialization_search_version="v1",
            deferral_initialization_search_max_steps=8000,
            deferral_trainer_version="v1",
            # Training configuration
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=2e-5,
            lr_scheduler_type="linear",
            warmup_ratio=0.04,
            weight_decay=0.0,
            # Logging and saving
            logging_steps=1,
            save_total_limit=1,
            bf16=True,
            # Additional arguments
            eval_strategy="no",
            save_strategy="epoch",
            report_to="wandb",
            logging_first_step=True,
            tf32=True,
        )
        phase1_trainer.run()

        logger.info("Phase 1 completed: Deferral token initialization saved.")

        # ========== Phase 2: Main Deferral Training with Marginal Likelihood Loss ==========
        logger.info("=" * 80)
        logger.info("Phase 2: Main deferral training with marginal likelihood loss...")
        logger.info("Using DeepSpeed ZeRO Stage 2 for memory optimization")
        logger.info("=" * 80)

        phase2_trainer = ModelTrainer(
            model_name_or_path=generator,
            tokenizer_name=generator,
            output_dir=f"{collm_dir}/model_checkpoints_final",
            dataset_name=f"{collm_dir}/mentor_scored_data",  # Use mentor scored data for phase 2
            # Model configuration
            use_fast_tokenizer=False,
            use_flash_attn=True,
            torch_dtype="bfloat16",
            max_seq_length=2048,
            preprocessing_num_workers=64,
            # Deferral configuration - load from phase 1
            no_deferral_initialization_search=False,  # Do not skip - we need to load gperp
            deferral_initialization_path=f"{collm_dir}/model_checkpoints_init/gperp_init.bin",  # Load from phase 1
            deferral_trainer_version="v1",
            # Training configuration
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,  # Enable for phase 2
            learning_rate=2e-5,
            lr_scheduler_type="linear",
            warmup_ratio=0.04,
            weight_decay=0.0,
            # Logging and saving
            logging_steps=1,
            save_total_limit=1,
            save_strategy="epoch",
            bf16=True,
            # Additional arguments
            eval_strategy="no",
            report_to="wandb",
            logging_first_step=True,
            tf32=True,
            overwrite_output_dir=True,
        )
        phase2_trainer.run()

        logger.info("=" * 80)
        logger.info("Co-LLM training completed successfully!")
        logger.info(f"Phase 1 checkpoints: {collm_dir}/model_checkpoints_init")
        logger.info(f"Phase 2 (final) checkpoints: {collm_dir}/model_checkpoints_final")
        logger.info("=" * 80)
    else:
        logger.info(f"Model checkpoints already exist at {collm_dir}/model_checkpoints_final. Skipping training.")

    # Run inference if requested
    if hyperparameters.get("run_inference", True):
        logger.info("\n" + "=" * 80)
        logger.info("Starting inference with trained model...")
        logger.info("=" * 80)

        # Get inference hyperparameters
        inference_split = hyperparameters.get("inference_split", "test")
        deferral_threshold = hyperparameters.get("deferral_threshold", 0.5)
        deferral_strategy = hyperparameters.get("deferral_strategy", "defer")
        max_tokens = hyperparameters.get("max_tokens", 512)
        threshold_warmup_schedule = hyperparameters.get("threshold_warmup_schedule", "none")
        threshold_warmup_steps = hyperparameters.get("threshold_warmup_steps", 15)
        save_inference_results = hyperparameters.get("save_inference_results", True)

        avg_score = run_inference(
            task=task,
            task_type=task_type,
            split=inference_split,
            model_base=f"{collm_dir}/model_checkpoints_final",
            model_ref=mentor,  # Use the original mentor model
            gpu_ids=gpu_ids,
            deferral_threshold=deferral_threshold,
            deferral_strategy=deferral_strategy,
            max_tokens=max_tokens,
            threshold_warmup_schedule=threshold_warmup_schedule,
            threshold_warmup_steps=threshold_warmup_steps,
            save_results=save_inference_results,
        )

        logger.info(f"\nFinal {inference_split} score: {avg_score:.4f}")

    return collm_dir
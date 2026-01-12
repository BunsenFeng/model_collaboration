#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    HfArgumentParser,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from model_collaboration.utils import collm_trainer as deferral_training_tools

logger = logging.getLogger(__name__)
DEFAULT_DEFERRAL_TOKEN = "<|defer|>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    no_deferral_initialization_search: bool = field(
        default=False,
        metadata={"help": "Whether to search for a good initialization for the deferral token."},
    )

    deferral_initialization_search_max_steps: Optional[int] = field(
        default=4000,
        metadata={"help": "The maximum number of steps to train to find the initialization for a deferral token."},
    )

    deferral_initialization_weight_balance: Optional[float] = field(
        default=None,
        metadata={"help": "The weight balance between the deferral token and the rest of the tokens."},
    )

    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use the Flash Attention module instead of the Llama Attention module."},
    )

    deferral_trainer_version: Optional[str] = field(
        default="v1",
        metadata={"help": "The version of the deferral trainer to use."},
    )

    deferral_initialization_search_version: Optional[str] = field(
        default="v1",
        metadata={"help": "The version of the initialization search to use."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json/jsonl file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."


class CoLLMTrainer:
    """
    A class for training models with deferral mechanism using Co-LLM approach.

    This class handles the complete training pipeline including:
    1. Model and tokenizer initialization
    2. Deferral token initialization search
    3. Training with custom deferral trainer
    """

    def __init__(
        self,
        model_name_or_path,
        output_dir,
        train_file=None,
        dataset_name=None,
        dataset_config_name=None,
        tokenizer_name=None,
        config_name=None,
        max_seq_length=None,
        max_train_samples=None,
        preprocessing_num_workers=None,
        cache_dir=None,
        use_fast_tokenizer=False,
        use_flash_attn=False,
        torch_dtype=None,
        no_deferral_initialization_search=False,
        deferral_initialization_search_max_steps=4000,
        deferral_initialization_weight_balance=None,
        deferral_trainer_version="v1",
        deferral_initialization_search_version="v1",
        streaming=False,
        overwrite_cache=False,
        training_args=None,
    ):
        """
        Initialize the CoLLMTrainer.

        Args:
            model_name_or_path: Path to pretrained model or model identifier
            output_dir: Directory to save the trained model
            train_file: Path to training file (json/jsonl)
            dataset_name: Name of the dataset to use
            dataset_config_name: Configuration name of the dataset
            tokenizer_name: Pretrained tokenizer name or path
            config_name: Pretrained config name or path
            max_seq_length: Maximum sequence length
            max_train_samples: Maximum number of training samples
            preprocessing_num_workers: Number of workers for preprocessing
            cache_dir: Cache directory for models
            use_fast_tokenizer: Whether to use fast tokenizer
            use_flash_attn: Whether to use flash attention
            torch_dtype: Torch dtype for model ("auto", "bfloat16", "float16", "float32")
            no_deferral_initialization_search: Skip deferral token initialization search
            deferral_initialization_search_max_steps: Max steps for initialization search
            deferral_initialization_weight_balance: Weight balance for deferral token
            deferral_trainer_version: Version of the deferral trainer
            deferral_initialization_search_version: Version of initialization search
            streaming: Enable streaming mode
            overwrite_cache: Overwrite cached datasets
            training_args: TrainingArguments object (if None, must call train() with it)
        """
        # Store model arguments
        self.model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            config_name=config_name,
            tokenizer_name=tokenizer_name,
            cache_dir=cache_dir,
            use_fast_tokenizer=use_fast_tokenizer,
            torch_dtype=torch_dtype,
            no_deferral_initialization_search=no_deferral_initialization_search,
            deferral_initialization_search_max_steps=deferral_initialization_search_max_steps,
            deferral_initialization_weight_balance=deferral_initialization_weight_balance,
            use_flash_attn=use_flash_attn,
            deferral_trainer_version=deferral_trainer_version,
            deferral_initialization_search_version=deferral_initialization_search_version,
        )

        # Store data arguments
        self.data_args = DataTrainingArguments(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            train_file=train_file,
            max_train_samples=max_train_samples,
            streaming=streaming,
            overwrite_cache=overwrite_cache,
            preprocessing_num_workers=preprocessing_num_workers,
            max_seq_length=max_seq_length,
        )

        # Store training arguments
        self.training_args = training_args
        self.output_dir = output_dir

        # Initialize later
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self, training_args=None):
        """
        Setup model, tokenizer, and datasets.

        Args:
            training_args: TrainingArguments object
        """
        if training_args is not None:
            self.training_args = training_args

        if self.training_args is None:
            raise ValueError("training_args must be provided either in __init__ or setup()")

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if self.training_args.should_log:
            transformers.utils.logging.set_verbosity_info()

        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training parameters {self.training_args}")

        # Set seed
        set_seed(self.training_args.seed)

        # Load config
        config_kwargs = {
            "cache_dir": self.model_args.cache_dir,
            "revision": self.model_args.model_revision,
            "use_auth_token": True if self.model_args.use_auth_token else None,
        }
        if self.model_args.config_name:
            config = AutoConfig.from_pretrained(self.model_args.config_name, **config_kwargs)
        elif self.model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, **config_kwargs)
        else:
            raise ValueError("You must provide either config_name or model_name_or_path")

        # Load tokenizer
        tokenizer_kwargs = {
            "cache_dir": self.model_args.cache_dir,
            "use_fast": self.model_args.use_fast_tokenizer,
            "revision": self.model_args.model_revision,
            "use_auth_token": True if self.model_args.use_auth_token else None,
        }
        if self.model_args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name, **tokenizer_kwargs)
        elif self.model_args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError("You must provide either tokenizer_name or model_name_or_path")

        # Load model
        if self.model_args.model_name_or_path:
            torch_dtype = (
                self.model_args.torch_dtype
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2" if self.model_args.use_flash_attn else "eager",
            )

        logger.info(f"Model loaded: {self.model.lm_head.weight.data.shape}")

        # Add special tokens
        if isinstance(self.tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            num_added_tokens = self.tokenizer.add_special_tokens(
                {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                }
            )
            assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token"
        elif isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            num_added_tokens = self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token"
        elif isinstance(self.tokenizer, GPT2Tokenizer) and isinstance(self.model, OPTForCausalLM):
            num_added_tokens = self.tokenizer.add_special_tokens({"unk_token": "<unk>"})

        if self.tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad_token. Adding '<pad>' as pad_token.")
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Resize embeddings
        deferral_training_tools.smart_tokenizer_and_embedding_resize(self.tokenizer, self.model)

        # Load dataset
        self.data_module = deferral_training_tools.make_supervised_data_module(
            tokenizer=self.tokenizer, data_args=self.data_args
        )

        # Search for deferral token initialization
        if self.model_args.no_deferral_initialization_search:
            logger.info("No deferral token initialization search requested, skipping.")
            gperp = None
        else:
            with self.training_args.main_process_first(desc="Search for deferral token initialization"):
                init_path = f"{self.training_args.output_dir}/gperp_init.bin"
                if os.path.isfile(init_path):
                    logger.info(f"Loading deferral token initialization from {init_path}")
                    gperp = torch.load(init_path)
                else:
                    logger.info("Searching for deferral token initialization")
                    search_deferral_initialization = deferral_training_tools.ALL_INITIALIZATION_SEARCH[
                        self.model_args.deferral_initialization_search_version
                    ]
                    gperp = search_deferral_initialization(
                        self.model,
                        self.data_module,
                        batch_size=8,
                        deferral_weight=self.model_args.deferral_initialization_weight_balance,
                        max_steps=None,
                    )
                    os.makedirs(self.training_args.output_dir, exist_ok=True)
                    torch.save(gperp, init_path)
                    logger.info(f"Saved deferral token initialization to {init_path}")

        # Add deferral token
        num_added_tokens = self.tokenizer.add_special_tokens(
            dict(additional_special_tokens=[DEFAULT_DEFERRAL_TOKEN])
        )
        assert num_added_tokens == 1, "Should only add one special token - the deferral token"

        deferral_training_tools.smart_tokenizer_and_embedding_resize_for_def_token(
            self.tokenizer, self.model, gperp
        )

        # Initialize trainer
        DeferralTrainer = deferral_training_tools.ALL_DEFERRAL_TRAINERS[self.model_args.deferral_trainer_version]
        self.trainer = DeferralTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            **self.data_module,
        )

    def train(self, resume_from_checkpoint=None):
        """
        Run the training process.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise ValueError("Must call setup() before train()")

        logger.info("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples
            if self.data_args.max_train_samples is not None
            else len(self.data_module["train_dataset"])
        )
        metrics["train_samples"] = min(max_train_samples, len(self.data_module["train_dataset"]))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        logger.info("Training completed!")
        return metrics

    def run(self, training_args=None, resume_from_checkpoint=None):
        """
        Complete training pipeline: setup and train.

        Args:
            training_args: TrainingArguments object
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        self.setup(training_args)
        return self.train(resume_from_checkpoint)

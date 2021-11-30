# Stuff
import os
from datasets.arrow_dataset import concatenate_datasets
import wandb
import random
import logging
import argparse
import numpy as np
from typing import List, Optional, Union, Dict

# PyTorch
import torch

# Hugging Face
import datasets
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from transformers.models.auto.configuration_auto import AutoConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
wandb.init(project="zeroshot-turkish", entity="emrecncelik")


def compute_metrics(eval_preds) -> Dict[str, float]:
    _accuracy = load_metric("accuracy")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = _accuracy.compute(predictions=predictions, references=labels)

    return accuracy


class NLITrainer:
    def __init__(
        self,
        checkpoint: str,
        dataset_name: Union[str, List[str]],
        validation_split: str,
        test_split: str,
        output_dir: Optional[str] = "",
        max_train_examples: Optional[int] = None,
        max_eval_examples: Optional[int] = None,
        random_state: int = 42,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        eval_steps: int = 1000,
        save_steps: int = 10000,
    ) -> None:
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        self.validation_split = validation_split
        self.test_split = test_split
        self.output_dir = output_dir
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.save_steps = save_steps

        self.raw_dataset: datasets.dataset_dict.DatasetDict
        self.tokenized_dataset: datasets.dataset_dict.DatasetDict
        self.model: transformers.AutoModelForSequenceClassification
        self.tokenizer: transformers.AutoTokenizer
        self.data_collator: transformers.DataCollatorWithPadding
        self.trainer: transformers.Trainer

        if not output_dir:
            self.output_dir = "_".join([checkpoint.replace("/", "_"), dataset_name])
        else:
            dir = "_".join([checkpoint.replace("/", "_"), dataset_name])
            self.output_dir = os.path.join(self.output_dir, dir)
        self.set_random_state(random_state)
        self.prepare_for_training()

    def set_random_state(self, random_state):
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)

    def load_dataset(self) -> None:
        logger.info(f"======== Loading dataset: {self.dataset_name} ========")
        if self.dataset_name in ["snli_tr", "multinli_tr"]:
            self.raw_dataset = load_dataset("nli_tr", self.dataset_name)

        elif self.dataset_name == "allnli_tr":
            # Concatenate two datasets
            raw_snli = load_dataset("nli_tr", "snli_tr")
            raw_multnli = load_dataset("nli_tr", "multinli_tr")
            self.raw_dataset = raw_snli

            self.raw_dataset["train"] = concatenate_datasets(
                [raw_snli["train"], raw_multnli["train"]]
            )
            self.raw_dataset["validation_mismatched"] = raw_multnli[
                "validation_mismatched"
            ]
            self.raw_dataset["validation_matched"] = raw_multnli["validation_matched"]

        else:
            raise ValueError(
                f"Dataset name {self.dataset_name} is not an option. Use 'mergenli_tr', 'snli_tr' or 'multinli_tr'."
            )

            self.raw_dataset = self.raw_dataset.filter(
                lambda example: example["label"] != -1
            )
            if self.max_train_examples:
                self.raw_dataset["train"] = self.raw_dataset["train"].select(
                    range(self.max_train_examples)
                )
            if self.max_eval_examples:
                self.raw_dataset[self.validation_split] = self.raw_dataset[
                    self.validation_split
                ].select(range(self.max_eval_examples))

        logger.info(f"\tTrain shape: {self.raw_dataset['train'].shape}")
        logger.info(
            f"\tValidation shape: {self.raw_dataset[self.validation_split].shape}"
        )
        logger.info(f"\tTest shape: {self.raw_dataset[self.test_split].shape}")

    def load_tokenizer(self) -> None:
        logger.info(
            f"======== Loading tokenizer from checkpoint: {self.checkpoint} ========"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def load_model(self) -> None:
        logger.info(
            f"======== Loading model from checkpoint: {self.checkpoint} ========"
        )
        self.config = AutoConfig.from_pretrained(
            self.checkpoint, num_labels=3, finetuning_task=self.dataset_name
        )
        self.config.label2id = {
            self.raw_dataset["train"].features["label"].int2str(id_): id_
            for id_ in range(3)
        }
        self.config.id2label = {v: k for k, v in self.config.label2id.items()}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, config=self.config
        )
        try:
            logger.info("\tApplying gradient checkpointing")
            self.model.gradient_checkpointing_enable()
        except ValueError:
            logger.info(
                f"\tGradient checkpointing not supported by model {self.checkpoint}"
            )

    def prepare_data(self) -> None:
        logger.info("======== Preparing the dataset for training ========")
        tokenize = lambda example: self.tokenizer(
            example["premise"], example["hypothesis"], truncation=True
        )
        tokenized_dataset = self.raw_dataset.map(tokenize, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset = tokenized_dataset.remove_columns(
            ["premise", "hypothesis", "idx"]
        )
        tokenized_dataset.set_format("torch")
        self.tokenized_dataset = tokenized_dataset
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def prepare_trainer(self) -> None:

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            seed=self.random_state,
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
        )

        self.trainer = Trainer(
            self.model,
            training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset[self.validation_split],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

    def prepare_for_training(self) -> None:
        self.load_dataset()
        self.load_tokenizer()
        self.prepare_data()

        self.load_model()
        self.prepare_trainer()

    def train(self):
        logger.info("======== Running training ========")
        train_result = self.trainer.train()
        metrics = train_result.metrics

        self.trainer.save_model()
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def evaluate(self):
        logger.info("======== Running evaluation ========")
        if self.dataset_name == "allnli_tr":
            for split in [
                "validation_matched",
                "validation_mismatched",
                "test",
                "validation",
            ]:
                metrics = self.trainer.evaluate(self.tokenized_dataset[split])
                self.trainer.log_metrics(split, metrics)
                self.trainer.save_metrics(split, metrics)

        else:
            metrics = self.trainer.evaluate()
            self.trainer.log_metrics(self.validation_split, metrics)
            self.trainer.save_metrics(self.validation_split, metrics)

            metrics = self.trainer.evaluate(self.tokenized_dataset[self.test_split])
            self.trainer.log_metrics(self.test_split, metrics)
            self.trainer.save_metrics(self.test_split, metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine-tune transformers on Turkish NLI data"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name or path of the model checkpoint",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Name or path of the dataset",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Training and evaluation batch size",
        required=False,
        default=32,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--max_train_examples",
        help="Maximum num. of training examples",
        required=False,
        default=None,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--max_eval_examples",
        help="Maximum num. of evaluation examples",
        required=False,
        default=None,
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory for model output",
        required=False,
        default="",
        type=str,
    )
    args = vars(parser.parse_args())

    if args["dataset"] == "snli_tr":
        setup = {
            "dataset": "snli_tr",
            "validation_split": "validation",
            "test_split": "test",
        }
    elif args["dataset"] == "multinli_tr":
        setup = {
            "dataset": "multinli_tr",
            "validation_split": "validation_matched",
            "test_split": "validation_mismatched",
        }
    elif args["dataset"] == "allnli_tr":
        setup = {
            "dataset": "allnli_tr",
            "validation_split": "validation_matched",
            "test_split": "validation_mismatched",
        }

    model = args["model"]

    # Start training
    trainer = NLITrainer(
        checkpoint=model,
        dataset_name=setup["dataset"],
        validation_split=setup["validation_split"],
        test_split=setup["test_split"],
        output_dir=args["output_dir"],
        batch_size=args["batch_size"],
        max_train_examples=args["max_train_examples"],
        max_eval_examples=args["max_eval_examples"],
    )

    trainer.train()
    trainer.evaluate()

from __future__ import annotations

# Python stuff
import os
import logging
import multiprocessing
from os.path import join
from typing import List, Dict, Union
from collections import Counter, OrderedDict

# Plotting
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Hugging Face
from datasets import load_dataset
from datasets import Dataset as HFDataset
from datasets.dataset_dict import DatasetDict as HFDatasetDict

from experiments.config import DATA_DIR, PLOTS_DIR
from experiments.preprocess import Preprocessor

from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# TODO Allow user to give labels for the dataset
# also set labels in the HFDataset  when loading from local csv.

font = {"family": "normal", "size": 12}
matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (8, 6)


class Dataset:
    dataset: HFDatasetDict

    def __init__(
        self,
        name: str,
        context: str,
        from_: str,
        text_col: str = "text",
        label_col: str = "label",
        text_preprocess: List[str] = ["remove_urls", "clean", "normalize_whitespace"],
        label_preprocess: List[str] = [],
        label_map: Union[bool, Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.context = context
        self.text_col = text_col
        self.label_col = label_col
        self.from_ = from_
        self.text_preprocess = text_preprocess
        self.label_preprocess = label_preprocess
        self.label_map = label_map
        self.stats = {}

        if text_preprocess:
            self.text_preprocessor = Preprocessor(steps=self.text_preprocess)
        if label_preprocess:
            self.label_preprocessor = Preprocessor(steps=self.label_preprocess)

    def __repr__(self) -> str:
        return f"{self.name}_{self.context}_{self.from_}"

    def __str__(self) -> str:
        return f"{self.name}_{self.context}_{self.from_}"

    def load_dataset(self):
        """Loads the dataset from local or from Hugging Face Hub.

        Returns:
            Dataset: Returns self to allow function aggregation
        """

        logger.info(f"======== Loading dataset {self.name} from {self.from_} ========")
        if self.from_ == "hf":
            self.dataset = load_dataset(self.name)
            self.dataset["train"] = self.dataset["train"].rename_columns(
                {self.text_col: "text", self.label_col: "label"}
            )
            if "test" in self.dataset:
                self.dataset["test"] = self.dataset["test"].rename_columns(
                    {self.text_col: "text", self.label_col: "label"}
                )
            else:
                logger.info(
                    f"\tDataset does not contain test split, call train_test_split"
                )
        elif self.from_ == "local":
            self.dataset = HFDatasetDict()
            for split in ["train", "test"]:
                file_path = join(DATA_DIR, self.name, f"formatted_{split}.csv")
                try:
                    dataset = HFDataset.from_csv(file_path)
                    dataset = dataset.rename_columns(
                        {self.text_col: "text", self.label_col: "label"}
                    )

                    self.dataset[split] = dataset
                except FileNotFoundError:
                    logger.info(
                        f"\tDataset does not contain formatted data for the {split} split."
                    )
                    logger.info(f"\tFile path: {file_path}")
                    logger.info(f"\tYou may need to call train_test_split function.")
        self.text_col = "text"
        self.label_col = "label"
        return self

    def train_test_split(self, test_size: float, random_state: int, **kwargs):
        """Splits data into train and test sets if data does not have
        a pre-determined test split

        Args:
            test_size (float): Percentage of test size
            random_state (int): Allows reproducibility

        Returns:
            Dataset: Returns self to allow function aggregation
        """
        if not "test" in self.dataset:
            self.dataset = self.dataset["train"].train_test_split(
                test_size=test_size, seed=random_state, **kwargs
            )
        return self

    def preprocess(self, column: str):
        """Applies preprocessing steps given in initialization. It can be applied to
        text or label columns.

        Args:
            column (str): can be "text" or "label"

        Returns:
            Dataset: Returns self to allow function aggregation
        """
        if column == "label":
            if not self.label_preprocess:
                return self
            steps = self.label_preprocess
            preprocessor = self.label_preprocessor
            column = self.label_col

        elif column == "text":
            if not self.text_preprocess:
                return self
            steps = self.text_preprocess
            preprocessor = self.text_preprocessor
            column = self.text_col

        logger.info(f"======== Preprocessing {column} ========")
        logger.info(f"\t{steps}")

        def _preprocess_col(example):
            example[column] = preprocessor.preprocess(example[column])
            return example

        logger.info(f"\tBefore:")
        for example in self.dataset["train"][column][:3]:
            logger.info(f"\t\t{str(example)[:20]}")

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                _preprocess_col,
                num_proc=multiprocessing.cpu_count(),
            )

        logger.info(f"\tAfter:")
        for example in self.dataset["train"][column][:3]:
            logger.info(f"\t\t{str(example)[:20]}")

        return self

    def map_labels(self):
        """Maps labels in the dataset to corresponding values
        in the self.label_map dictionary.

        Returns:
            Dataset: Returns self to allow function aggregation
        """
        if self.label_map:
            logger.info(f"{self}")
            logger.info(f"\tBefore:")
            for example in self.dataset["train"][self.label_col][:3]:
                logger.info(f"\t\t{str(example)[:20]}")

            if isinstance(self.label_map, bool) and self.label_map:
                map_func = self.dataset["train"].features[self.label_col].int2str

            elif isinstance(self.label_map, dict):
                map_func = lambda x: self.label_map[str(x)]

            def _map_label(example):
                example[self.label_col] = map_func(example[self.label_col])
                return example

            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    _map_label, num_proc=multiprocessing.cpu_count()
                )

            logger.info(f"\tAfter:")
            for example in self.dataset["train"][self.label_col][:3]:
                logger.info(f"\t\t{str(example)[:20]}")

        return self

    def compute_dataset_statistics(self):
        self._get_dataset_size()
        self._compute_label_counts()
        self._compute_token_counts()
        return self

    def _get_dataset_size(self):
        counts = {"train": 0, "test": 0, "total": 0}
        for split in ["train", "test"]:
            if split in self.dataset:
                counts[split] += len(self.dataset[split])

        counts["total"] = counts["test"] + counts["train"]
        self.stats["dataset_size"] = counts

    def _compute_label_counts(self) -> None:
        for split in ["train", "test"]:
            if split in self.dataset:
                counter = Counter(self.dataset[split]["label"])
                counter = OrderedDict(sorted(counter.items()))
                self.stats[f"label_counts_{split}"] = counter

    def _plot_label_counts(self) -> None:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        labels, values = zip(*self.stats["label_counts_train"].items())
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color="tab:orange", height=0.25, label="train")

        labels, values = zip(*self.stats["label_counts_test"].items())
        ax.barh(y_pos + 0.25, values, color="tab:blue", height=0.25, label="test")

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Label")
        # ax.set_title(f"Label distribution of {self.name}")
        ax.set_yticks(y_pos + 0.125, labels=labels)
        ax.legend()

        save_dir = os.path.join(PLOTS_DIR, "label_counts")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        fig.savefig(
            os.path.join(save_dir, f"{self.name}_labels.png"),
            bbox_inches="tight",
        )

    def _compute_token_counts(self):
        counter = Counter()
        for split in ["train", "test"]:
            token_counts = []
            if split in self.dataset:
                for text in self.dataset[split]["text"]:
                    tokenized = word_tokenize(text)
                    token_counts.append(len(tokenized))
                    counter.update(tokenized)

                self.stats[f"token_counts_{split}"] = token_counts

        self.stats["total_tokens"] = sum(
            [sum(self.stats[key]) for key in self.stats.keys() if "token_counts" in key]
        )
        self.stats["total_unique_tokens"] = len(counter)

    def _plot_token_counts_distribution(self):
        pass

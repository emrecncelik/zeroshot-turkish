from __future__ import annotations

# Python stuff
import string
import os
import pickle
from loguru import logger
import multiprocessing
from os.path import join
from typing import List, Dict, Union
from collections import Counter, OrderedDict

# Plotting
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Hugging Face
from datasets import Value, load_dataset
from datasets import Dataset as HFDataset
from datasets.dataset_dict import DatasetDict as HFDatasetDict

from zeroshot_classification.config import DATA_DIR, PLOTS_DIR
from zeroshot_classification.preprocess import Preprocessor

from nltk.tokenize import word_tokenize


# TODO Allow user to give labels for the dataset
# also set labels in the HFDataset  when loading from local csv.

font = {"family": "normal", "size": 14}
matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (10, 8)

def filter_only_punct_num_or_empty(text):
    text = text.strip()
    if text == "" or all([True if t in string.punctuation or t.isdigit() else False for t in text]):
        return False
    return True

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
        self.dataset_dir = join(DATA_DIR, self.name)

        if text_preprocess:
            self.text_preprocessor = Preprocessor(steps=self.text_preprocess)
        if label_preprocess:
            self.label_preprocessor = Preprocessor(steps=self.label_preprocess)

    def __repr__(self) -> str:
        return f"{self.name}_{self.context}_{self.from_}"

    def __str__(self) -> str:
        return f"{self.name}_{self.context}_{self.from_}"

    @property
    def labels(self):
        lookup = set()
        labels = []
        if "test" in self.dataset:
            labels += self.dataset["test"][self.label_col]
        if "train" in self.dataset:
            labels += self.dataset["train"][self.label_col]

        labels = [l for l in labels if l not in lookup and lookup.add(l) is None]
        return labels

    def load_dataset(self):
        """Loads the dataset from local or from Hugging Face Hub.

        Returns:
            Dataset: Returns self to allow function aggregation
        """

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
                logger.warning(
                    f"Dataset does not contain test split, call train_test_split"
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
                    logger.warning(
                        f"Dataset does not contain formatted data for the {split} split."
                    )
                    logger.warning(f"File path: {file_path}")
                    logger.warning(f"You may need to call train_test_split function.")
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

        logger.debug(f"Preprocessing {column} column")
        logger.debug(f"\t{steps}")

        def _preprocess_col(example):
            example[column] = preprocessor.preprocess(example[column])
            return example

        logger.debug(f"\tBefore:")
        for example in self.dataset["train"][column][:3]:
            logger.debug(f"\t\t{str(example)[:70]}")

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                _preprocess_col,
                num_proc=multiprocessing.cpu_count(),
            )

        logger.debug(f"\tAfter:")
        for example in self.dataset["train"][column][:3]:
            logger.debug(f"\t{str(example)[:70]}")
      
        return self

    def filter(self, n_examples: int = None):
        self.dataset = self.dataset.filter(filter_only_punct_num_or_empty, input_columns=["text"])
        if n_examples:
            self.dataset["train"] = self.dataset["train"].select(range(n_examples))
            if "test" in self.dataset:
                self.dataset["test"] = self.dataset["test"].select(range(n_examples))
            
        return self

    def map_labels(self):
        """Maps labels in the dataset to corresponding values
        in the self.label_map dictionary.

        Returns:
            Dataset: Returns self to allow function aggregation
        """
        if self.label_map:
            logger.debug(f"{self}")
            logger.debug(f"\tBefore:")
            for example in self.dataset["train"][self.label_col][:3]:
                logger.debug(f"\t\t{str(example)[:20]}")

            if isinstance(self.label_map, bool) and self.label_map:
                map_func = self.dataset["train"].features[self.label_col].int2str

            elif isinstance(self.label_map, dict):
                map_func = lambda x: self.label_map[str(x)]

            def _map_label(example):
                example[self.label_col] = map_func(example[self.label_col])
                return example

            for split in self.dataset.keys():
                # Quick n dirty solution
                try:
                    self.dataset[split] = self.dataset[split].map(
                        _map_label, num_proc=multiprocessing.cpu_count()
                    )
                except ValueError:
                    new_features = self.dataset[split].features.copy()
                    new_features[self.label_col] = Value("string")
                    self.dataset[split] = self.dataset[split].cast(new_features)
                    self.dataset[split] = self.dataset[split].map(
                        _map_label, num_proc=multiprocessing.cpu_count()
                    )

            logger.debug(f"\tAfter:")
            for example in self.dataset["train"][self.label_col][:3]:
                logger.debug(f"\t\t{str(example)[:20]}")

        return self

    def compute_dataset_statistics(self, save: bool = True, load: bool = False):
        stats_filepath = join(self.dataset_dir, f"{self.name}_stats.pkl")
        if load:
            with open(stats_filepath, "rb") as stats_file:
                self.stats = pickle.load(stats_file)
        else:
            self._get_dataset_size()
            self._compute_label_counts()
            self._compute_token_counts()

        if save:
            with open(stats_filepath, "wb") as stats_file:
                pickle.dump(self.stats, stats_file)
        return self

    def plot_dataset_statistics(self, save: bool = True):
        self._plot_label_counts(save)
        self._plot_token_counts_distribution(save)
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

    def _plot_label_counts(self, save: bool) -> None:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        labels, values = zip(*self.stats["label_counts_train"].items())
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color="tab:orange", height=0.25, label="train")

        labels, values = zip(*self.stats["label_counts_test"].items())
        ax.barh(y_pos + 0.25, values, color="tab:blue", height=0.25, label="test")

        ax.set_xlabel("Label count")
        ax.set_ylabel("Label name")
        ax.set_yticks(y_pos + 0.125, labels=labels)
        ax.legend()

        if save:
            self.__save_plot(fig, "label_dist")
        else:
            return fig

    def _compute_token_counts(self):
        counter = Counter()
        for split in ["train", "test"]:
            token_counts = []
            counter_split = Counter()
            if split in self.dataset:
                for text in self.dataset[split]["text"]:
                    tokenized = word_tokenize(text)
                    token_counts.append(len(tokenized))
                    counter.update(tokenized)
                    counter_split.update(tokenized)

                self.stats[f"token_counts_{split}"] = token_counts
                self.stats[f"unique_token_counts_{split}"] = len(counter_split)
                self.stats[f"mean_token_counts_{split}"] = np.mean(token_counts)
                self.stats[f"std_token_counts_{split}"] = np.std(token_counts)

        self.stats["total_tokens"] = sum(
            [
                sum(self.stats[key])
                for key in self.stats.keys()
                if "token_counts_train" == key or "token_counts_test" == key
            ]
        )
        self.stats["total_unique_tokens"] = len(counter)

    def _plot_token_counts_distribution(self, save: bool):
        train = self.stats["token_counts_train"]
        test = self.stats["token_counts_test"]
        colors = ["tab:orange", "tab:blue"]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.hist([train, test], color=colors)
        n, bins, patches = ax1.hist([train, test], bins=50)
        ax1.cla()

        width = (bins[1] - bins[0]) * 0.4
        bins_shifted = bins + width
        ax1.bar(bins[:-1], n[0], width, align="edge", color=colors[0], label="train")
        ax2.bar(
            bins_shifted[:-1], n[1], width, align="edge", color=colors[1], label="test"
        )

        ax1.set_ylabel("Number of samples", color=colors[0])
        ax2.set_ylabel("Number of samples", color=colors[1])
        ax1.tick_params("y", colors=colors[0])
        ax2.tick_params("y", colors=colors[1])
        ax1.set_xlabel("Word count")
        ax1.legend(loc=2)
        ax2.legend(loc=1)

        plt.tight_layout()
        plt.show()

        if save:
            self.__save_plot(fig, "token_dist")
        else:
            return fig

    def __save_plot(
        self,
        fig: matplotlib.figure.Figure,
        suffix: str,
    ):
        filepath = os.path.join(PLOTS_DIR, f"{self.name}_{suffix}.png")
        if not os.path.exists(PLOTS_DIR):
            os.mkdir(PLOTS_DIR)

        fig.savefig(
            filepath,
            bbox_inches="tight",
        )

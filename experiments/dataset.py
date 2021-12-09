# Python stuff
import logging
from os.path import join

# Hugging Face
from datasets import load_dataset
from datasets import Dataset as HFDataset
from datasets.dataset_dict import DatasetDict as HFDatasetDict

from experiments.config import DATA_PATH


logger = logging.getLogger(__name__)


class Dataset:
    dataset: HFDatasetDict

    def __init__(
        self,
        name: str,
        task: str,
        from_: str,
        text_col: str = "text",
        label_col: str = "label",
    ) -> None:
        self.name = name
        self.task = task
        self.text_col = text_col
        self.label_col = label_col
        self.from_ = from_

    def __repr__(self) -> str:
        return f"{self.name}_{self.task}_{self.from_}"

    def __str__(self) -> str:
        return f"{self.name}_{self.task}_{self.from_}"

    def load_dataset(self):
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
                file_path = join(DATA_PATH, self.name, f"formatted_{split}.csv")
                try:
                    dataset = HFDataset.from_csv(file_path)
                    dataset = dataset.rename_columns(
                        {self.text_col: "text", self.label_col: "label"}
                    )
                    self.text_col = "text"
                    self.label_col = "label"
                    self.dataset[split] = dataset
                except FileNotFoundError:
                    logger.info(
                        f"\tDataset does not contain formatted data for the {split} split."
                    )
                    logger.info(f"\tFile path: {file_path}")
                    logger.info(f"\tYou may need to call train_test_split function.")
        return self

    def train_test_split(self, test_size: float, random_state: int, **kwargs):
        if not "test" in self.dataset:
            self.dataset = self.dataset["train"].train_test_split(
                test_size=test_size, seed=random_state, **kwargs
            )
        return self

    def compute_statistics(self, split: str = "train"):
        pass

    def compute_label_distribution(self, split: str = "train"):
        pass

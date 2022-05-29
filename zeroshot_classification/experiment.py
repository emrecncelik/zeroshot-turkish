from collections import defaultdict
from typing import Dict, List, Optional

import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

from zeroshot_classification.classifiers import (
    MLMZeroshotClassifier,
    NSPZeroshotClassifier,
    NLIZeroshotClassifier,
)
from zeroshot_classification.config import DATASETS, TEMPLATES, MODELS
from zeroshot_classification.dataset import Dataset
from zeroshot_classification.utils import serialize


class Experiment:
    name_to_cls = {
        "nli": NLIZeroshotClassifier,
        "nsp": NSPZeroshotClassifier,
        # "nsp": MLMZeroshotClassifier,
    }

    def __init__(
        self,
        model_type: str,
        datasets: Optional[Dict] = DATASETS,
        templates: Optional[Dict] = TEMPLATES,
        random_state: Optional[int] = 7,
        models: Optional[List[str]] = None,
        **model_kwargs,
    ) -> None:
        self.model_type = model_type
        self.datasets = datasets
        self.templates = templates
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.models = models if models else MODELS[model_type]

        self.current_classifier = None
        self.current_dataset = None

    def _initialize_classifier(self, model_name: str, **model_kwargs):
        if self.current_classifier is not None:
            logger.info(f"Deleting previous classifier.")
            self._empty_memory()

        logger.info("Initalizing classifier.")
        kwargs = model_kwargs if model_kwargs else self.model_kwargs
        self.current_classifier = self.name_to_cls[self.model_type](
            model_name=model_name, random_state=self.random_state, **kwargs
        )

    def run(
        self,
        **prediction_kwargs,
    ):
        results = defaultdict(
            lambda: defaultdict(dict),
        )
        linebreak = "=" * 60

        logger.info(f"Experiment running for {self.model_type.upper()} " + linebreak)

        for model_name in self.models:
            logger.info(f"Initializing current model: {model_name} " + linebreak)
            self._initialize_classifier(model_name, **self.model_kwargs)
            for dataset_kwargs in self.datasets:
                logger.info(
                    f"Loading current dataset: {dataset_kwargs['name']} " + linebreak
                )
                self._load_dataset(dataset_kwargs)
                for template in self.templates[dataset_kwargs["context"]]:
                    logger.info(
                        f"Predictions with the template: {template} " + linebreak
                    )
                    self.current_dataset = self.current_classifier.predict_on_dataset(
                        self.current_dataset,
                        candidate_labels=self.current_dataset.labels,
                        prompt_template=template,
                        **prediction_kwargs,
                    )

                    results[model_name][dataset_kwargs["name"]][
                        template
                    ] = self._evaluate_on_current_dataset()
        results.default_factory = None
        return results

    def _empty_memory(self):
        self.current_classifier = None
        torch.cuda.empty_cache()

    def _load_dataset(self, dataset: Dict):
        dataset = dataset.copy()
        if self.current_dataset is not None:
            logger.info(f"Deleting previous dataset: {self.current_dataset}")
            self.current_dataset = None

        test_size = dataset.pop("test_size")
        self.current_dataset = (
            Dataset(**dataset)
            .load_dataset()
            .map_labels()
            .preprocess("text")
            .preprocess("label")
            .train_test_split(test_size=test_size, random_state=self.random_state)
        )

    def _evaluate_on_current_dataset(self):
        true = self.current_dataset.dataset["test"]["label"]
        pred = self.current_dataset.dataset["test"]["predicted_label"]
        clf_report = classification_report(
            true,
            pred,
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(true, pred)
        acc = accuracy_score(true, pred)

        logger.info(
            "\n"
            + classification_report(
                true,
                pred,
                zero_division=0,
                output_dict=False,
            )
        )
        logger.info(f"Accuracy: {acc}")

        return {
            "classification_report": clf_report,
            "confusion_matrix": cm,
            "accuracy": acc,
        }


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # nli_experiment = Experiment("nli", device=0)
    # results = nli_experiment.run(batched=True, batch_size=100, num_workers=4)

    # serialize(results, "results_nli.bin")

    nsp_experiment = Experiment("nsp")
    results = nsp_experiment.run(batched=True, batch_size=10, num_workers=4)

    serialize(results, "results_nsp.bin")

# reform = {
#     (outerKey, innerKey): values
#     for outerKey, innerDict in results.items()
#     for innerKey, values in innerDict.items()
# }
# pd.DataFrame.from_dict(reform, orient="index").transpose()
# data.applymap(lambda x: x["classification_report"]["weighted avg"]["f1-score"] if isinstance(x, dict) else x)

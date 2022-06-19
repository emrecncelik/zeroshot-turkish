from collections import defaultdict
from typing import Dict, List, Optional

import torch
import wandb
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from zeroshot_classification.classifiers import (
    NSPZeroshotClassifier,
    NLIZeroshotClassifier,
    MLMZeroshotClassifier,
)
from zeroshot_classification.config import DATASETS, TEMPLATES, MODELS
from zeroshot_classification.dataset import Dataset
from zeroshot_classification.utils import serialize


class Experiment:
    name_to_cls = {
        "nli": NLIZeroshotClassifier,
        "nsp": NSPZeroshotClassifier,
        "mlm": MLMZeroshotClassifier,
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
        cache_results: Optional[str] = None,
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

                    cm_plot = results[model_name][dataset_kwargs["name"]][template].pop(
                        "confusion_matrix_plot"
                    )

                    with wandb.init(
                        project="zeroshot-turkish-predictions", entity="emrecncelik"
                    ) as run:
                        run.log(
                            {
                                "model": model_name,
                                "model_type": self.model_type,
                                "dataset": dataset_kwargs["name"],
                                "template": template,
                                "classification_report": results[model_name][
                                    dataset_kwargs["name"]
                                ][template]["classification_report"],
                                "confusion_matrix": cm_plot.figure_,
                            }
                        )
                        run.finish()

                    if cache_results:
                        logger.info(f"Caching results at {cache_results}")
                        results.default_factory = None
                        serialize(results, cache_results)
                        results.default_factory = lambda: defaultdict(dict)

        results.default_factory = None
        return results

    def _empty_memory(self):
        if self.model_type == "mlm":
            del self.current_classifier.ft_model_or_path
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
        cm_plot = ConfusionMatrixDisplay.from_predictions(true, pred)
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
            "confusion_matrix_plot": cm_plot,
            "accuracy": acc,
        }


if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # nli_experiment = Experiment("nli", device=0)
    # results = nli_experiment.run(
    #     cache_results="nli_results_cache.bin",
    #     batched=True,
    #     batch_size=256,
    #     num_workers=8,
    # )
    # del nli_experiment
    # serialize(results, "nli_results_final.bin")

    # nsp_experiment = Experiment("nsp")
    # results = nsp_experiment.run(
    #     cache_results="nsp_results_cache.bin",
    #     batched=True,
    #     batch_size=256,
    #     num_workers=8,
    # )
    # del nsp_experiment
    # serialize(results, "nsp_results_final.bin")

    mlm_experiment = Experiment("mlm")
    results = mlm_experiment.run(
        cache_results="mlm_results_cache.bin",
        batched=True,
        batch_size=128,
        num_workers=0,
    )
    del mlm_experiment
    serialize(results, "mlm_results_final.bin")

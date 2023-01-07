from collections import defaultdict
from typing import Dict, List, Optional

import time
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
from zeroshot_classification.config import DATASETS, TEMPLATES, MODELS, ROOT_DIR
from zeroshot_classification.dataset import Dataset
from zeroshot_classification.utils import serialize

# To control logging level for various modules used in the application:
import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

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
        log_to_wandb: bool = True,
        **prediction_kwargs,
    ):
        results = defaultdict(
            lambda: defaultdict(dict),
        )
        linebreak = "=" * 60

        current_exp = 0
        total_experiments = len(self.models) * len(self.datasets) * len(self.templates)
        start_time = time.time()
        total_time = 0
        logger.info(f"Experiment running for {self.model_type.upper()} " + linebreak)
        logger.info(f"Total of {total_experiments} experiments will be runned.")

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
                    if log_to_wandb:
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
                    end_time = time.time()
                    elapsed = (end_time - start_time) / 60 / 60
                    total_time += elapsed
                    current_exp += 1
                    logger.info(f"Experiment finished: {current_exp}/{total_experiments}")
                    logger.info(f"Elapsed time for experiment w/ template: {elapsed} hours")
                    if cache_results:
                        logger.info(f"Caching results at {cache_results}")
                        results.default_factory = None
                        serialize(results, cache_results)
                        results.default_factory = lambda: defaultdict(dict)


        logger.info(f"Elapsed time for whole experiment: {total_time} hours")
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
            .filter()
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
    import os
    import sys
    import fasttext

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    set_global_logging_level(prefices=["huggingface", "transformers", "datasets", "torch", "tensorflow", "fastText"])

    nli_experiment = Experiment("nli", device="gpu")
    results = nli_experiment.run(
        cache_results="nli_results_cache_rev.bin",
        log_to_wandb=False,
        batched=True,
        batch_size=16,
        num_workers=4,
    )
    del nli_experiment
    serialize(results, "nli_results_final_rev.bin")

    nsp_experiment = Experiment("nsp")
    results = nsp_experiment.run(
        cache_results="nsp_results_cache_rev.bin",
        log_to_wandb=False,
        batched=True,
        batch_size=16,
        num_workers=4,
    )
    del nsp_experiment
    serialize(results, "nsp_results_final_rev.bin")
    
    ft_model = fasttext.load_model(os.path.join(ROOT_DIR, "cc.tr.300.bin"))
    mlm_experiment = Experiment(
        "mlm",
        ft_model=ft_model,
    )

    results = mlm_experiment.run(
        cache_results="mlm_results_cache_rev.bin",
        log_to_wandb=False,
        batched=True,
        batch_size=16,
        num_workers=4,
    )
    del mlm_experiment
    serialize(results, "mlm_results_rev.bin")

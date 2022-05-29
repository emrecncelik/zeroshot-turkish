import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from loguru import logger
from transformers import (
    BertForNextSentencePrediction,
    BertTokenizer,
    BertTokenizerFast,
    pipeline,
)
from datasets import Dataset as HFDataset
from zeroshot_classification.dataset import Dataset
from zeroshot_classification.config import device


class ZeroshotClassifierBase(ABC):
    def __init__(self, model_name: str, random_state: int = None, **kwargs) -> None:
        self.model_name = model_name
        self.random_state = random_state

        self._set_random_state(random_state)
        self._init_model(model_name)

    @abstractmethod
    def _init_model(self, model_name: str):
        """_summary_

        Args:
            model_name (str): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def predict_on_dataset(
        self,
        Dataset: List[str],
        candidate_labels: List[str],
        prompt_template: str,
        **kwargs,
    ):
        """_summary_

        Args:
            Dataset (List[str]): _description_
            candidate_labels (List[str]): _description_
            prompt_template (str): _description_
        """

    @abstractmethod
    def predict_on_texts(
        self,
        texts: List[str],
        candidate_labels: List[str],
        prompt_template: str,
        **kwargs,
    ):
        """_summary_

        Args:
            texts (List[str]): _description_
            candidate_labels (List[str]): _description_
            prompt_template (str): _description_
        """

    def _set_random_state(self, random_state: int):
        if random_state:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

    # def _set_prompt_template(prompt_template: Union[str, List[str]]) -> str:
    #     if isinstance(prompt_template, list):
    #         prompt_template = random.choice(prompt_template)

    #     return prompt_template

    # def _prepare_prompts(
    #     self, candidate_labels: List[str], prompt_template: str
    # ) -> List[str]:
    #     return [prompt_template.format(label) for label in candidate_labels]


class NLIZeroshotClassifier(ZeroshotClassifierBase):
    def __init__(self, model_name: str, random_state: int = None, **kwargs) -> None:
        self.model_name = model_name
        self.random_state = random_state

        self._set_random_state(random_state)
        self._init_model(model_name, **kwargs)

    def _init_model(self, model_name: str, **kwargs):
        self.model = pipeline("zero-shot-classification", model=model_name, **kwargs)

    def predict_on_dataset(
        self,
        dataset: Dataset,
        candidate_labels: List[str],
        prompt_template: str,
        batched: bool = True,
        batch_size: int = 100,
        **kwargs,
    ):
        def _predict(examples):
            texts = examples["text"]
            outputs = self.model(
                texts,
                candidate_labels=candidate_labels,
                hypothesis_template=prompt_template,
                **kwargs,
            )
            labels = []
            for output in outputs:
                labels.append(output["labels"][0])

            assert len(labels) == len(texts)
            examples["predicted_label"] = labels
            return examples

        dataset.dataset["test"] = dataset.dataset["test"].map(
            _predict, batched=batched, batch_size=batch_size
        )

        return dataset

    def predict_on_texts(
        self,
        texts: List[str],
        candidate_labels: List[str],
        prompt_template: str,
        **kwargs,
    ):
        outputs = self.model(
            texts,
            candidate_labels=candidate_labels,
            hypothesis_template=prompt_template,
            **kwargs,
        )
        for out in outputs:
            del out["sequence"]

        return outputs


class NSPZeroshotClassifier(ZeroshotClassifierBase):
    def __init__(
        self,
        model_name: str,
        random_state: int = None,
        reverse_prompts: bool = False,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.random_state = random_state
        self.reverse_prompts = reverse_prompts
        self._set_random_state(random_state)
        self._init_model(model_name, **kwargs)

    def _init_model(self, model_name: str, **kwargs):
        self.model = BertForNextSentencePrediction.from_pretrained(
            model_name, **kwargs
        ).to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def _prepare_prompts(self, candidate_labels: List[str], prompt_template: str):
        self.labels = candidate_labels
        self.label2prompt = {
            label: prompt_template.replace("{}", label) for label in candidate_labels
        }
        self.prompt2label = {v: k for k, v in self.label2prompt.items()}
        logger.info(f"Prompts:")
        logger.info(self.label2prompt)

    def predict_on_dataset(
        self,
        dataset: Dataset,
        candidate_labels: List[str],
        prompt_template: str,
        batched: bool = True,
        batch_size: int = 100,
        **kwargs,
    ):
        self._prepare_prompts(candidate_labels, prompt_template)

        def _predict(examples):
            """This is the worst piece of code I have ever written"""
            texts = examples["text"]
            prompts = list(self.label2prompt.values())
            predictions = []

            for text in texts:
                prompt_isNext_logits = []
                for prompt in prompts:
                    if self.reverse_prompts:
                        encoding = self.tokenizer(
                            prompt,
                            text,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                        )
                    else:
                        encoding = self.tokenizer(
                            text,
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                        )
                    outputs = self.model(**encoding.to(device))
                    prompt_isNext_logits.append(
                        outputs.logits.cpu().detach().numpy()[0][0]
                    )

                predicted_prompt_index = np.argmax(prompt_isNext_logits)
                predicted_class = self.prompt2label[prompts[predicted_prompt_index]]
                predictions.append(predicted_class)

            examples["predicted_label"] = predictions

            return examples

        dataset.dataset["test"] = dataset.dataset["test"].map(
            _predict, batched=batched, batch_size=batch_size
        )

        return dataset

    def predict_on_texts(
        self,
        texts: List[str],
        candidate_labels: List[str],
        prompt_template: str,
        **kwargs,
    ):
        pass
        # outputs = self.model(
        #     texts,
        #     candidate_labels=candidate_labels,
        #     hypothesis_template=prompt_template,
        #     **kwargs,
        # )
        # for out in outputs:
        #     del out["sequence"]

        # return outputs


class MLMZeroshotClassifier(ZeroshotClassifierBase):
    pass

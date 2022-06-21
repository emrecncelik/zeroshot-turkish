import random
from abc import ABC, abstractmethod
from typing import List

import fasttext
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from scipy.spatial.distance import cosine
from scipy.special import softmax
from torch.utils.data import Dataset as PTDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForNextSentencePrediction,
    BertTokenizerFast,
    pipeline,
)
from transformers.pipelines import FillMaskPipeline, base

from zeroshot_classification.config import device
from zeroshot_classification.dataset import Dataset


class ZeroshotMLMDataset(PTDataset):
    def __init__(self, texts: List[str], template: str, mask_token: str):
        self.texts = texts
        self.template = template.replace("{}", mask_token)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx] + " " + self.template


class FillMaskPipelineWithPreprocessParams(FillMaskPipeline):
    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters):
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(
            inputs, return_tensors=return_tensors, **preprocess_parameters
        )
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

    def _sanitize_parameters(self, top_k=None, targets=None, **tokenizer_kwargs):
        postprocess_params = {}
        preprocess_params = tokenizer_kwargs

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise base.PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                "The tokenizer does not define a `mask_token`.",
            )
        return preprocess_params, {}, postprocess_params


class ZeroshotClassifierBase(ABC):
    def __init__(self, model_name: str, random_state: int = None, **kwargs) -> None:
        self.model_name = model_name
        self.random_state = random_state

        self._set_random_state(random_state)
        self._init_model(model_name)

    @abstractmethod
    def _init_model(self, model_name: str):
        pass

    @abstractmethod
    def predict_on_dataset(
        self,
        dataset: Dataset,
        candidate_labels: List[str],
        prompt_template: str,
        batched: bool = True,
        batch_size: int = 100,
        **kwargs,
    ):
        pass

    @abstractmethod
    def predict_on_texts(
        self,
        texts: List[str],
        candidate_labels: List[str],
        prompt_template: str,
        **kwargs,
    ):
        pass

    def _set_random_state(self, random_state: int):
        if random_state:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)


class NLIZeroshotClassifier(ZeroshotClassifierBase):
    def __init__(self, model_name: str, random_state: int = None, **kwargs) -> None:
        self.model_name = model_name
        self.random_state = random_state

        self._set_random_state(random_state)
        self._init_model(model_name, **kwargs)

    def __init__(self, model_name: str, random_state: int = None, **kwargs) -> None:
        super().__init__(model_name, random_state, **kwargs)

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
        super().__init__(model_name, random_state, **kwargs)
        self.reverse_prompts = reverse_prompts

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
        self._prepare_prompts(candidate_labels, prompt_template)
        prompts = list(self.label2prompt.values())
        predictions = []

        for text in texts:
            logits = []
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
                logits.append(
                    outputs.logits.cpu().detach().numpy()[0][0]
                )

            predicted_labels = list(self.label2prompt.keys())
            predicted_probs = softmax(logits).tolist()
            print("before")
            print(predicted_labels)
            print(predicted_probs)
            print("after")
            predicted_probs, predicted_labels = zip(*sorted(zip(predicted_probs, predicted_labels), reverse=True))
            print(predicted_labels)
            print(predicted_probs)
            predictions.append(
                {
                "labels": predicted_labels,
                "probabilities": predicted_probs
                }
            )

        return predictions


class MLMZeroshotClassifier(ZeroshotClassifierBase):
    def __init__(
        self,
        model_name: str,
        ft_model: str = None,
        ft_path: str = None,
        random_state: int = None,
        **kwargs,
    ) -> None:
        self.ft_model = ft_model
        self.ft_path = ft_path

        if ft_model is None and ft_path is None:
            raise ValueError("Provide at least one of ft_model or ft_path.")

        super().__init__(model_name, random_state, **kwargs)

    def _init_model(self, model_name: str):
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(0)
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")

        pipeline = FillMaskPipelineWithPreprocessParams(
            model=model, tokenizer=tokenizer, device=0
        )

        self.mask_token = tokenizer.mask_token
        self.model = pipeline
        if self.ft_model is None:
            logger.info("Initializing fastText model")
            self.ft_model = fasttext.load_model(self.ft_path)

    def predict_on_dataset(
        self,
        dataset: Dataset,
        candidate_labels: List[str],
        prompt_template: str,
        batched: bool = True,
        batch_size: int = 100,
        **kwargs,
    ):
        if not batched:
            batch_size = 1

        candidate_label_vectors = [
            self.ft_model.get_sentence_vector(label) for label in candidate_labels
        ]
        pipeline = self.model(
            ZeroshotMLMDataset(
                dataset.dataset["test"]["text"],
                prompt_template,
                mask_token=self.mask_token,
            ),
            truncation=True,
            batch_size=batch_size,
            **kwargs,
        )

        predictions = []
        for model_output in tqdm(pipeline, total=len(dataset.dataset["test"]["text"])):
            avg_vector = self.ft_model.get_sentence_vector(
                " ".join([o["token_str"] for o in model_output])
            )
            similarities = [
                1 - cosine(avg_vector, vec) for vec in candidate_label_vectors
            ]
            pred = sorted(
                zip(candidate_labels, similarities),
                key=lambda tup: tup[1],
                reverse=True,
            )
            logger.debug(f"Prediction: {pred[0][0]}")
            predictions.append(pred[0][0])

        try:
            dataset.dataset["test"] = dataset.dataset["test"].add_column(
                "predicted_label", predictions
            )
        except ValueError:
            dataset.dataset["test"] = dataset.dataset["test"].remove_columns(
                ["predicted_label"]
            )
            dataset.dataset["test"] = dataset.dataset["test"].add_column(
                "predicted_label", predictions
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

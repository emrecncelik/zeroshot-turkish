import torch
import random
import numpy as np
from abc import abstractmethod
from typing import List, Union


class ZeroshotClassifierBase:
    def __init__(self, model_name: str, random_state: int = None) -> None:
        self.model_name = model_name
        self.random_state = random_state

        self._set_random_state(random_state)
        self._init_model(model_name)

    def _set_random_state(random_state: int):
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        np.seed(random_state)
        random.seed(random_state)

    @abstractmethod
    def _init_model(self, model_name: str):
        raise NotImplementedError

    def _set_template(template: Union[str, List[str]]) -> str:
        if isinstance(template, list):
            template = random.choice(template)

        return template

    @abstractmethod
    def predict(self, text: str, labels: List[str], template: Union[str, List[str]]):
        raise NotImplementedError

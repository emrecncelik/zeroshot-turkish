import re
import string
from toolz import functoolz
from typing import List
from textacy import preprocessing
from experiments.utils import mild_cleaning
from turkish.deasciifier import Deasciifier


class Preprocessor:
    def __init__(self, steps: List[str]) -> None:
        self.steps = steps
        self._name2func = {
            "remove_urls": self.remove_urls,
            "normalize_whitespace": self.normalize_whitespace,
            "deasciify": self.deasciify,
            "lower": self.lower,
            "upper": self.upper,
            "remove_punct": self.remove_punct,
            "clean": mild_cleaning,
        }

    @property
    def preprocess(self):
        steps = [
            self._name2func[step] if isinstance(step, str) else step
            for step in self.steps
        ]
        return functoolz.compose_left(*steps)

    def remove_urls(self, text: str) -> str:
        text = preprocessing.replace.urls(text)
        return text.replace("_URL_", " ")

    def normalize_whitespace(self, text: str) -> str:
        return " ".join(text.split())

    def deasciify(self, text: str) -> str:
        return Deasciifier(text).convert_to_turkish()

    def lower(self, text: str) -> str:
        text = re.sub(r"İ", "i", text)
        text = re.sub(r"I", "ı", text)
        text = text.lower()
        return text

    def upper(self, text: str) -> str:
        text = re.sub(r"i", "İ", text)
        text = text.upper()
        return text

    def remove_punct(self, text: str) -> str:
        # Not so fast but OK
        for punct in string.punctuation:
            text = text.replace(punct, " ")
        return text

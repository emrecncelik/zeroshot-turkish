import os
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOT_DIR = os.getenv("ZEROSHOT_ROOT_DIR", "/home/bilgi/emrecan/zeroshot-turkish")
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots_revision")

MODELS = {
    "nli": [
        "emrecan/distilbert-base-turkish-cased-allnli_tr",
        # "emrecan/distilbert-base-turkish-cased-multinli_tr",
        # "emrecan/distilbert-base-turkish-cased-snli_tr",
        "emrecan/bert-base-turkish-cased-allnli_tr",
        # "emrecan/bert-base-turkish-cased-multinli_tr",
        # "emrecan/bert-base-turkish-cased-snli_tr",
        "emrecan/convbert-base-turkish-mc4-cased-allnli_tr",
        # "emrecan/convbert-base-turkish-mc4-cased-multinli_tr",
        # "emrecan/convbert-base-turkish-mc4-cased-snli_tr",
        "emrecan/bert-base-multilingual-cased-allnli_tr",
        # "emrecan/bert-base-multilingual-cased-multinli_tr",
        # "emrecan/bert-base-multilingual-cased-snli_tr",
    ],
    "nsp": [
        "dbmdz/bert-base-turkish-cased",
        # "dbmdz/bert-base-turkish-uncased",
        "dbmdz/bert-base-turkish-128k-cased",
        # "dbmdz/bert-base-turkish-128k-uncased",

        # Additional @ Rev. 1
        "bert-base-multilingual-cased",
    ],
    "mlm": [
        "dbmdz/bert-base-turkish-cased",
        # "dbmdz/bert-base-turkish-uncased",
        "dbmdz/bert-base-turkish-128k-cased",
        # "dbmdz/bert-base-turkish-128k-uncased",

        # Revision 1
        "dbmdz/convbert-base-turkish-cased",

        # Additional @ Rev. 1
        # "bert-base-multilingual-cased",
        # "dbmdz/distilbert-base-turkish-cased",
        # "distilbert-base-multilingual-cased",
        # "xlm-roberta-base",
    ],
}


TEMPLATES = {
    "emotion_or_sentiment": [
        # "Bu metnin içerdiği duygu {}",
        "Bu metnin içerdiği duygu çoğunlukla {}",
        # "Bu metin {} duygular içeriyor",
        "Bu metin çoğunlukla {} duygular içeriyor",
        # "{} duygular hissediyorum",
        "Çoğunlukla {} duygular hissediyorum",
    ],
    "news": [
        # "Bu haberin konusu {}",
        "Bu haberin konusu çoğunlukla {}",
        # "Bu haber {} ile ilgilidir",
        "Bu haber çoğunlukla {} ile ilgilidir",
        # "Bu haberin içeriği {} ile ilgilidir",
        "Bu haberin içeriği çoğunlukla {} ile ilgilidir",
    ],
    "review": [
        # "Bu şikayetin konusu {}",
        "Bu şikayetin konusu çoğunlukla {}",
        # "{} ile ilgili şikayetim var",
        # "Çoğunlukla {} ile ilgili şikayetim var",
        "{} hizmetinizden memnun değilim",
        # "Çoğunlukla {} hizmetinizden memnun değilim",
        # "Bu şikayetin içeriği {} ile ilgili",
        "Bu şikayetin içeriği çoğunlukla {} ile ilgili",
    ],
}

DATASETS = [
    {
        "name": "turkish_movie_reviews_big",
        "context": "emotion_or_sentiment",
        "from_": "local",
        "label_col": "label",
        "test_size": 0.3,
    },
    {
        "name": "turkish_movie_reviews_small",
        "context": "emotion_or_sentiment",
        "from_": "local",
        "label_col": "label",
        "test_size": 0.3,
    },
    {
        "name": "ttc3600",
        "context": "news",
        "from_": "local",
        "label_col": "label",
        "label_map": {
            "ekonomi": "ekonomi",
            "kultursanat": "kültürsanat",
            "saglik": "sağlık",
            "siyaset": "siyaset",
            "spor": "spor",
            "teknoloji": "teknoloji",
        },
        "test_size": 0.3,
    },
    {
        "name": "ttc4900",
        "context": "news",
        "from_": "local",
        "label_col": "category",
        "label_preprocess": ["deasciify"],
        "test_size": 0.3,
    },
    {
        "name": "17bintweet",
        "context": "emotion_or_sentiment",
        "from_": "local",
        "label_preprocess": ["deasciify"],
        "test_size": 0.3,
    },
    {
        "name": "tremo",
        "context": "emotion_or_sentiment",
        "from_": "local",
        "label_map": {
            "Happy": "mutluluk",
            "Fear": "korku",
            "Sadness": "üzüntü",
            "Surprise": "şaşırma",
            "Disgust": "tiksinme",
            "Anger": "öfke",
            "Ambigious": "belirsiz",
        },
        "test_size": 0.3,
    },
    {
        "name": "ruh_hali",
        "context": "emotion_or_sentiment",
        "from_": "local",
        "label_map": {
            "karisik": "karışık",
            "neseli": "neşeli",
            "sinirli": "sinirli",
            "uzgun": "üzgün",
        },
        "test_size": 0.3,
    },

    # {
    #     "name": "turted",
    #     "context": "emotion_or_sentiment",
    #     "from_": "local",
    #     "test_size": 0.3,
    # },
    {
        "name": "turkish_product_reviews",
        "context": "emotion_or_sentiment",
        "from_": "hf",
        "label_col": "sentiment",
        "text_col": "sentence",
        "label_map": {"1": "olumlu", "0": "olumsuz"},
        "test_size": 0.3,
    },
    {
        "name": "tc32",
        "context": "review",
        "from_": "local",
        "label_col": "category",
        "label_preprocess": ["deasciify", "remove_punct"],
        "test_size": 0.3,
    },
    # {
    #     "name": "offenseval2020_tr",
    #     "context": "emotion_or_sentiment",
    #     "from_": "hf",
    #     "label_col": "subtask_a",
    #     "text_col": "tweet",
    #     "label_map": {"0": "saldırgan olmayan", "1": "saldırgan"},
    # },
    # {
    #     "name": "interpress_news_category_tr",
    #     "context": "news",
    #     "from_": "hf",
    #     "label_col": "category",
    #     "text_col": "content",
    #     "label_preprocess": ["deasciify"],
    # },
]

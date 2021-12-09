import os

DATA_PATH = os.getenv("DATA_PATH", "/home/emrecan/tez/zeroshot-turkish/datasets")

NLI_MODELS = [
    "emrecan/distilbert-base-turkish-cased-allnli_tr",
    "emrecan/distilbert-base-turkish-cased-multinli_tr",
    "emrecan/distilbert-base-turkish-cased-snli_tr",
    "emrecan/bert-base-turkish-cased-allnli_tr",
    "emrecan/bert-base-turkish-cased-multinli_tr",
    "emrecan/bert-base-turkish-cased-snli_tr",
    "emrecan/convbert-base-turkish-mc4-cased-allnli_tr",
    "emrecan/convbert-base-turkish-mc4-cased-multinli_tr",
    "emrecan/convbert-base-turkish-mc4-cased-snli_tr",
    "emrecan/bert-base-multilingual-cased-allnli_tr",
    "emrecan/bert-base-multilingual-cased-multinli_tr",
    "emrecan/bert-base-multilingual-cased-snli_tr",
]

NSP_MODELS = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/bert-base-turkish-uncased",
    "dbmdz/bert-base-turkish-128k-cased",
    "dbmdz/bert-base-turkish-128k-uncased",
]


TEMPLATES = {
    "emotion": [
        "Bu metnin içerdiği duygu {}",
        "Bu metnin içerdiği duygu çoğunlukla {}",
        "Bu metin {} duygular içeriyor",
        "Bu metnin çoğunlukla {} duygular içeriyor",
        "{} duygular hissediyorum",
        "Çoğunlukla {} duygular hissediyorum",
    ],
    "news": [
        "Bu haberin konusu {}",
        "Bu haberin konusu çoğunlukla {}",
        "Bu haber {} ile ilgilidir",
        "Bu haber çoğunlukla {} ile ilgilidir",
        "Bu haberin içeriği {} ile ilgilidir",
        "Bu haberin içeriği çoğunlukla {} ile ilgilidir",
    ],
    "review": [
        "Bu şikayetin konusu {}",
        "Bu şikayetin konusu çoğunlukla {}",
        "{} ile ilgili şikayetim var",
        "Çoğunlukla {} ile ilgili şikayetim var",
        "{} hizmetinizden memnun değilim",
        "Çoğunlukla {} hizmetinizden memnun değilim",
        "Bu şikayetin içeriği {} ile ilgili",
        "Bu şikayetin içeriği çoğunlukla {} ile ilgili",
    ],
}

DATASETS = [
    {
        "name": "17bintweet",
        "task": "sentiment",
        "from_": "local",
    },
    {
        "name": "tc32",
        "task": "review",
        "from_": "local",
        "label_col": "category",
    },
    {
        "name": "ttc3600",
        "task": "news",
        "from_": "local",
        "label_col": "label",
    },
    {
        "name": "ttc4900",
        "task": "news",
        "from_": "local",
        "label_col": "category",
    },
    {
        "name": "ruh_hali",
        "task": "emotion",
        "from_": "local",
    },
    {
        "name": "offenseval2020_tr",
        "task": "other",
        "from_": "hf",
        "label_col": "subtask_a",
        "text_col": "tweet",
    },
    {
        "name": "interpress_news_category_tr",
        "task": "other",
        "from_": "hf",
        "label_col": "category",
        "text_col": "content",
    },
    {
        "name": "turkish_product_reviews",
        "task": "sentiment",
        "from_": "hf",
        "label_col": "sentiment",
        "text_col": "sentence",
    },
]

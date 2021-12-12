import os
import logging
import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "/home/emrecan/tez/zeroshot-turkish/datasets")


def format_txt_dataset(
    dir_to_folders: str,
    encoding: str = None,
    clean: bool = True,
):
    data = {"file_name": [], "text": [], "label": []}
    for root, _, files in os.walk(dir_to_folders):
        txt_files = [file_ for file_ in files if file_.endswith(".txt")]
        if not txt_files:
            continue

        label = root.split("/")[-1].strip().lower()
        for file_ in txt_files:
            with open(os.path.join(root, file_), "r", encoding=encoding) as f:
                text = f.read()
            data["label"].append(label)
            data["text"].append(text)
            data["file_name"].append(file_)

    data = pd.DataFrame(data)
    if clean:
        data["text"] = data["text"].apply(mild_cleaning)
    data.to_csv(os.path.join(dir_to_folders, "formatted.csv"), index=False)


def mild_cleaning(text):
    mistakes = {
        "&ccedil": "ç",
        "&uuml;": "ü",
        "&ouml;": "ö",
        "\n": " ",
        "\t": " ",
        "﻿": " ",
        "...Devamını oku": " ",
    }
    for key, value in mistakes.items():
        text = " ".join(text.replace(key, value).strip().split()).strip()

    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    format_txt_dataset(os.path.join(DATA_PATH, "ruh_hali"), encoding="cp1254")
    format_txt_dataset(os.path.join(DATA_PATH, "ttc3600"), encoding="utf-8")

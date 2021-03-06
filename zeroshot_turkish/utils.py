import pickle
import os
import logging
from typing import Any
import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "/home/emrecan/tez/zeroshot-turkish/datasets")


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


def format_tremo(filepath: str):
    data = pd.read_xml(filepath)
    data = data[["Entry", "ValidatedEmotion"]]
    data.columns = ["text", "label"]

    filename = filepath.split("/")
    filename[-1] = "formatted_train.csv"
    filename = "/".join(filename)
    data.to_csv(filename, index=False)


def serialize(obj: Any, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def deserialize(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def format_results(result_path: str, metric: str = "accuracy", averaging: str = None):
    # nli = format_results('/home/emrecan/tez/zeroshot-turkish/results/big/nli_results_final.bin', "accuracy")
    # nsp = format_results('/home/emrecan/tez/zeroshot-turkish/results/big/nsp_results_final.bin', "accuracy")
    # df = nli.join(nsp).round(3)

    results = deserialize(result_path)
    # [‘index’, ‘columns’, ‘data’, ‘index_names’, ‘column_names’].
    columns = []
    index = []
    data = []
    index_names = ["dataset", "template"]
    column_names = ["model"]

    for model, dict1 in results.items():
        columns.append(model)
        for dataset, dict2 in dict1.items():
            for template, dict3 in dict2.items():
                if (dataset, template) not in index:
                    index.append((dataset, template))
                    if metric in dict3["classification_report"]:
                        data.append([dict3["classification_report"][metric]])
                    else:
                        data.append([dict3["classification_report"][averaging][metric]])
                else:
                    if metric in dict3["classification_report"]:
                        data[index.index((dataset, template))].append(
                            dict3["classification_report"][metric]
                        )
                    else:
                        data[index.index((dataset, template))].append(
                            dict3["classification_report"][averaging][metric]
                        )

    return pd.DataFrame.from_dict(
        {
            "columns": columns,
            "index": index,
            "data": data,
            "column_names": column_names,
            "index_names": index_names,
        },
        orient="tight",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # format_txt_dataset(os.path.join(DATA_PATH, "ruh_hali"), encoding="cp1254")
    # format_txt_dataset(os.path.join(DATA_PATH, "ttc3600"), encoding="utf-8")
    format_tremo(os.path.join(DATA_PATH, "tremo", "TREMODATA.xml"))

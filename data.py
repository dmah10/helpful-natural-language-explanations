import numpy as np
import pandas as pd
import json
from datasets import Dataset, Features, Value
from datasets import disable_progress_bar

COLUMN_NAMES = {
    "esnli": {
        "premise": "text_1",
        "hypothesis": "text_2",
    },
    "HFC": {
        "en_claim": "text_1",
        "en_filtered_sentences": "text_2",
    },
}


def df_to_hf(df):
    disable_progress_bar()
    dataset = Dataset.from_pandas(
        df,
        features=Features(
            {
                "text_1": Value("string"),
                "text_2": Value("string"),
                "label": Value("int32"),
                "explanation": Value("string"),
                "human_explanation": Value("string"),
            }
        ),
        preserve_index=False,
    ).class_encode_column("label")

    return dataset


def load_dataset(
    name,
    explanation_model,
    explanation_type,
    base_path="data",
    test_size=0.2,
    split_idx=0,
):
    col_replace_names = COLUMN_NAMES[name]
    train_file = f"{name}_{explanation_model}_{explanation_type}_{split_idx}_train.csv"
    test_file = f"{name}_{explanation_model}_{explanation_type}_{split_idx}_test.csv"
    train_df = pd.read_csv(f"{base_path}/{train_file}")
    test_df = pd.read_csv(f"{base_path}/{test_file}")
    train_df.rename(columns=col_replace_names, inplace=True)
    test_df.rename(columns=col_replace_names, inplace=True)
    trainset = df_to_hf(train_df)
    testset = df_to_hf(test_df)

    return trainset, testset


def load_dataset_to_split(
    name,
    explanation_model,
    explanation_type,
    base_path="data",
    test_size=0.2,
):
    col_replace_names = COLUMN_NAMES[name]
    if "esnli" in name:
        name = "esnli_balanced_final"
    elif "HFC" in name:
        name = "HFC_final"
    filename = f"explanations_{explanation_model}_{name}_{explanation_type}.csv"
    df = pd.read_csv(f"{base_path}/{filename}")
    df.rename(columns=col_replace_names, inplace=True)
    return df

import numpy as np
import pandas as pd
import random

random.seed(42)
from datetime import datetime

# Labels
ai_label = "<|AI|>"
lg_label = "<|ML|>"
cl_label = "<|CL|>"
cv_label = "<|CV|>"
endoftext = "<|endoftext|>"
sep = "<|sep|>"


def read_datasets(subject, filename):
    """Read a .tsv dataset and extracts titles, abstracts, and created dates

    Args:
        subject: A label from ("<|AI|>", "<|ML|>", "<|CL|>", "<|CV|>").
        filename: The name of the dataset file.

    Returns:
        A list of zipped titles, abstracts, and created dates
    """
    dataset = pd.read_csv(filename, delimiter="\t")
    titles = [f"{subject} {title} {sep}" for title in list(dataset["title"])]
    abstracts = list(dataset["abstract"])
    date = [datetime.strptime(i, "%Y-%m-%d") for i in list(dataset["created"])]
    return list(zip(titles, abstracts, date))


def merge_datasets():
    """Merge all four datasets.

    Returns:
        A list of titles, abstracts, dates, sorted by dates.
    """
    ai = read_datasets(ai_label, "cs.AI.tsv")
    lg = read_datasets(lg_label, "cs.LG.tsv")
    cl = read_datasets(cl_label, "cs.CL.tsv")
    cv = read_datasets(cv_label, "cs.CV.tsv")
    data = ai + lg + cl + cv
    return sorted(data, key=lambda x: x[-1])


def split_datasets(data):
    """Split the dataset into train, valid, test sets.

    Args:
        data: A list of titles, abstracts, dates, sorted by dates.

    Returns:
        Train, valid, test sets.
    """
    train_text = data[:-12000]
    eval_text = data[-12000:]
    valid_test_ratio = 0.5
    valid_text = eval_text[: int(len(eval_text) * valid_test_ratio)]
    test_text = eval_text[int(len(eval_text) * valid_test_ratio) :]
    assert len(train_text) == 109616
    assert len(valid_text) == 6000
    assert len(test_text) == 6000
    return (train_text, valid_text, test_text)


def write_datasets(data, name):
    """Write a .txt file of the dataset

    Args:
        data: A list of titles, abstracts, dates, sorted by dates.
        name: The name of the file to write.
    """
    with open(name + ".txt", "w+") as f:
        for d in data:
            f.write(f"{d[0]} {d[1]}{endoftext}\n\n")
    f.close()
    print(f"{name} file completed.")


if __name__ == "__main__":
    data = merge_datasets()
    train, valid, test = split_datasets(data)
    write_datasets(train, "train")
    write_datasets(valid, "valid")
    write_datasets(test, "test")

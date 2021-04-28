import random
import re
from datetime import datetime

import pandas as pd

random.seed(42)

# Labels
startoftext = "<|startoftext|>"
endoftext = "<|endoftext|>"
sep = "<|sep|>"
path = "data/"


def read_datasets(filename):
    """Read a .tsv dataset and extracts titles, abstracts, and created dates

    Args:
        filename: The name of the dataset file.

    Returns:
        A list of zipped titles, abstracts, and created dates
    """
    dataset = pd.read_csv(filename, delimiter="\t")
    titles = [f"{startoftext} {title} {sep}" for title in list(dataset["title"])]
    abstracts = [abst + endoftext for abst in list(dataset["abstract"])]
    date = [datetime.strptime(i, "%Y-%m-%d") for i in list(dataset["created"])]
    arxiv_id = [re.sub("[^0-9]", "", i) for i in list(dataset["arxiv_id"])]
    return list(zip(titles, abstracts, arxiv_id, date))


def merge_datasets():
    """Merge all four datasets.

    Returns:
        A list of titles, abstracts, dates, sorted by dates.
    """
    ai = read_datasets(path + "cs.AI.tsv")
    lg = read_datasets(path + "cs.LG.tsv")
    cl = read_datasets(path + "cs.CL.tsv")
    cv = read_datasets(path + "cs.CV.tsv")
    data = ai + lg + cl + cv
    unique_ids = set()
    filtered_data = []
    for d in data:
        if d[-2] not in unique_ids:
            unique_ids.add(d[-2])
            filtered_data.append(d)
    return sorted(filtered_data, key=lambda x: x[-1])


def split_datasets(data):
    """Split the dataset into train, valid, test sets.

    Args:
        data: A list of titles, abstracts, dates, sorted by dates.

    Returns:
        Train, valid, test sets.
    """
    train_text = data[:-9880]
    eval_text = data[-9880:]
    valid_test_ratio = 0.5
    valid_text = eval_text[: int(len(eval_text) * valid_test_ratio)]
    test_text = eval_text[int(len(eval_text) * valid_test_ratio) :]
    assert len(train_text) == 90000
    assert len(valid_text) == 4940
    assert len(test_text) == 4940
    return (train_text, valid_text, test_text)


def write_datasets(data, name):
    """Write a .txt file of the dataset

    Args:
        data: A list of titles, abstracts, dates, sorted by dates.
        name: The name of the file to write.
    """
    with open(path + name + ".txt", "w+") as f:
        for d in data:
            f.write(f"{d[0]} {d[1]}\n\n")
    f.close()
    print(f"{name} file completed.")


if __name__ == "__main__":
    data = merge_datasets()
    train, valid, test = split_datasets(data)
    write_datasets(train, "train")
    write_datasets(valid, "valid")
    write_datasets(test, "test")

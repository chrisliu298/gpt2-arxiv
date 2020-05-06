import numpy as np
import pandas as pd
import random

# random.seed(42)


ai_label = "<|ai|>"
lg_label = "<|lg|>"
cl_label = "<|cl|>"
cv_label = "<|cv|>"
title_label = "<|title|>"
abstract_label = "<|abstract|>"
endoftext = "<|endoftext|>"


def make_dataset(subject, titles, abstract):
    data = list(zip(titles, abstract))
    random.shuffle(data)
    with open(f"{subject[2:4]}_dataset.txt", "w+") as f:
        for t, a in data:
            f.write(
                f"{subject} {title_label} {t} \n\n{abstract_label} {a} {endoftext}\n\n"
            )
    f.close()


def main():
    ai_dataset = pd.read_csv("cs.AI.tsv", delimiter="\t")
    lg_dataset = pd.read_csv("cs.LG.tsv", delimiter="\t")
    cl_dataset = pd.read_csv("cs.CL.tsv", delimiter="\t")
    cv_dataset = pd.read_csv("cs.CV.tsv", delimiter="\t")

    ai_titles = list(ai_dataset["title"])
    ai_abstract = list(ai_dataset["abstract"])
    lg_titles = list(lg_dataset["title"])
    lg_abstract = list(lg_dataset["abstract"])
    cl_titles = list(cl_dataset["title"])
    cl_abstract = list(cl_dataset["abstract"])
    cv_titles = list(cv_dataset["title"])
    cv_abstract = list(cv_dataset["abstract"])

    make_dataset(ai_label, ai_titles, ai_abstract)
    make_dataset(lg_label, lg_titles, lg_abstract)
    make_dataset(cl_label, cl_titles, cl_abstract)
    make_dataset(cv_label, cv_titles, cv_abstract)


if __name__ == "__main__":
    main()

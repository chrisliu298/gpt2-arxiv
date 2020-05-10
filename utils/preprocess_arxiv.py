import numpy as np
import pandas as pd
import random
random.seed(42)

ai_label = "<|AI|>"
lg_label = "<|ML|>"
cl_label = "<|CL|>"
cv_label = "<|CV|>"
endoftext = "<|endoftext|>"
sep = "<|sep|>"


def make_dataset(subject, titles, abstract):
    data = list(zip(titles, abstract))
    random.shuffle(data)
    text = []
    with open(f"{subject[2:4]}_dataset.txt", "w+") as f:
        for title, abstract in data:
            example = f"{subject} {title} {sep} {abstract} {endoftext}\n\n"
            f.write(example)
            text.append(example)
    f.close()
    return text


def main():
    ai_dataset = pd.read_csv("arxiv_dataset/cs.AI.tsv", delimiter="\t")
    lg_dataset = pd.read_csv("arxiv_dataset/cs.LG.tsv", delimiter="\t")
    cl_dataset = pd.read_csv("arxiv_dataset/cs.CL.tsv", delimiter="\t")
    cv_dataset = pd.read_csv("arxiv_dataset/cs.CV.tsv", delimiter="\t")

    ai_titles = list(ai_dataset["title"])
    ai_abstract = list(ai_dataset["abstract"])
    lg_titles = list(lg_dataset["title"])
    lg_abstract = list(lg_dataset["abstract"])
    cl_titles = list(cl_dataset["title"])
    cl_abstract = list(cl_dataset["abstract"])
    cv_titles = list(cv_dataset["title"])
    cv_abstract = list(cv_dataset["abstract"])

    ai_text = make_dataset(ai_label, ai_titles, ai_abstract)
    lg_text = make_dataset(lg_label, lg_titles, lg_abstract)
    cl_text = make_dataset(cl_label, cl_titles, cl_abstract)
    cv_text = make_dataset(cv_label, cv_titles, cv_abstract)

    print(f"AI paper count: {len(ai_dataset)}")
    print(f"ML paper count: {len(lg_dataset)}")
    print(f"CL paper count: {len(cl_dataset)}")
    print(f"CV paper count: {len(cv_dataset)}")
    print(f"All paper count: {len(ai_dataset) + len(lg_dataset) + len(cl_dataset) + len(cv_dataset)}")

    arxiv_dataset = ai_text + lg_text + cl_text + cv_text
    random.shuffle(arxiv_dataset)

    ratio = 0.9

    arxiv_train = arxiv_dataset[:int(len(arxiv_dataset) * ratio)]
    arxiv_eval = arxiv_dataset[int(len(arxiv_dataset) * ratio):]

    with open("arxiv_train.txt", "w+") as train:
        for i in arxiv_train:
            train.write(i)
    train.close()

    with open("arxiv_eval.txt", "w+") as eval:
        for i in arxiv_eval:
            eval.write(i)
    eval.close()



if __name__ == "__main__":
    main()

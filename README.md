# CSE142 Project6: Fine-Tuning GPT-2 to Generate Research Paper Abstracts

This GPT-2 model is capable of generating abstracts given paper titles.

## ArXiv Dataset

This repo contains a part of the dataset from [arxiv_archive](https://github.com/staeiou/arxiv_archive). Specifically, it contains papers under aritficial intelligence (AI), machine learning (LG), computation and language (CL), and computer vision and pattern recognition (CV) categories. 

I use the titles and abstract of these papers to fine-tune my GPT-2 model.

|   Splits   |   Count    | Percentage (%) | BPE Token Count |
| :--------: | :--------: | :------------: | :-------------: |
|   Train    |   90,000   |     90.11      |   20,834,012    |
| Validation |    4,940   |      4.95      |    1,195,056    |
|    Test    |    4,940   |      4.95      |    1,218,754    |
| **Total**  | **99,880** |    **100**     | **23,247,822*** |

**Note:** The two extra tokens here are the `\n`'s at the bottom of the file.

### Usage

The script reads the `.tsv` file, adds special symbols to separate the titile and the abstract, sorts them by dates, and writes a text file for each category and an extra file of shuffled instances. It splits all examples into training, validation, and test sets.

```shell
> python preprocess_arxiv.py
```


## Getting Started

### Prerequisites

```shell
> pip install -r requirements.txt
```

## Fine-Tune

```shell
> python train.py
```

## Generate

```shell
> python generate.py
```

## Reference

```
@dataset{r_stuart_geiger_2019_2533436,
    author= {R. Stuart Geiger},
    title={{ArXiV Archive: A tidy and complete archive of metadata for papers on arxiv.org, 1993-2019}},
    month=jan,
    year= 2019,
    publisher={Zenodo},
    version= {v1.0.1},
    doi={10.5281/zenodo.2533436},
    url={https://doi.org/10.5281/zenodo.2533436}
}
```


## Author(s)

- Chris Liu


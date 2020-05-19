# CSE142 Project6: GPT-2-arXiv

This GPT-2 model is capable of generating abstracts given paper titles.

We will add argparse support in later versions.

## ArXiv Dataset

This repo contains a part of the dataset from [arxiv_archive](https://github.com/staeiou/arxiv_archive). Specifically, it contains papers under aritficial intelligence (AI), machine learning (LG), computation and language (CL), and computer vision and pattern recognition (CV) categories. 

I use the titles and abstract of these papers to fine-tune my GPT-2 model.

| Category  |   Count    | Percentage (%) | BPE Token Count |
| :-------: | :--------: | :------------: | :-------------: |
|    AI     |   21889    |     18.00      |    4,791,146    |
|    LG     |   47025    |     38.67      |   11,078,662    |
|    CL     |   17008    |     13.99      |    3,549,625    |
|    CV     |   35694    |     29.35      |    8,687,225    |
| **Total** | **121616** |    **100**     | **28,106,658**  |

|   Splits   |   Count    | Percentage (%) | BPE Token Count |
| :--------: | :--------: | :------------: | :-------------: |
|   Train    |   109616   |     90.13      |   25,201,566    |
| Validation |    6000    |      4.93      |    1,435,898    |
|    Test    |    6000    |      4.93      |    1,469,196    |
| **Total**  | **121616** |    **100**     | **28,106,660*** |

**Note:** The two extra tokens here are the `\n`'s at the bottom of the file.

### Usage

The script reads the `.tsv` file, adds special symbols to separate the titile and the abstract, and writes a text file for each category and an extra file of shuffled instances. It splits all examples into training, validation, and test sets.

```shell
python preprocess_arxiv.py
```


## Getting Started

### Prerequisites

```
> pip install -r requirements.txt
```

## Fine-Tune

```
> python train.py
```

## Generate

```
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


# CLS-Coding-Exercise

## Introduction

This project is part of a coding interview at the Canadian Light Source.

The tasks include: <br>
<ul>
    <li>Question 1:  Given the following dataset containing ~1.5 million rows of data (question1-data.csv.gz) and no other information, what meaningful information can you learn from the data? We would like to see whatever code you come up with and also a summary of the results. Use as much or as little of the data as is reasonable.</li><br>
    <li> Question 2: Given the following Mathematical expression (question2-equation.png) which represents the Mutual Information between two random variables, write a Python function to calculate the Discrete Mutual Information between two random variables. A test dataset is provided (question2-data.h5), with 4 columns of data for 4 random variables (“a”, “b”, “x”, “y”). You can use your function to calculate the mutual information between pairs of variables, for example, “a” and “x”.</li>
</ul>


## Table of Contents
- [Introduction](#introduction)
- [Project Organization](#project-organization)
- [Installation](#installation)
- [Usage](#usage)



## Project Organization

In order to helps maintaining a consistent, organized, and efficient workflow, this project uses the following template:

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Installation

This project uses <strong>Makefile</strong> to automate the build and management processes.

### Getting Started

Open your terminal and navigate to the project's root directory where the<strong>Makefile</strong> is located. Execute the following commands based on your needs:

1. Setting up the environment:

```bash
make setup-environment
```
>Note: This command installs all dependencies listed in the environment.yml file located in the root directory. Update this YAML file to add or remove packages.

2. Installing local modules for development:

```bash
make setup-package
```
>Note: This command install the src directory as a package, allowing you to utilize the current state of the code within src throughout this environment.

3. To analyze the data

```bash
make run-pipeline
```

>Note: This command triggers the main.py located in the root directory, facilitating data analysis and providing answers to either question 1 or 2, based on your preference. Configure your choice using the config.yml file in the root directory.

## Usage

### 1. Data

Due to confidentiality and size concerns, data is typically not pushed to GitHub. Before executing the pipeline or notebooks, ensure you place the datasets in the <strong><em>/data/raw</em></strong> directories.

### 2. Main Pipeline

To choose the pipeline for question 1 or question 2, we need to change the <strong>current_question</strong> parameter in the config.yml.

#### For Example: To choose a pipeline for question 2

```bash
question_1:
    data_path: data\raw\question1_data.csv
    chunksize: 500000
    loggine: True

question_2:
    data_path: data\raw\question2-data.h5
    mutual_information_function: mutual_information_binary
    # List of available functions for calculating discrete mutual information:
    # [mutual_information_binary, mutual_information_multiple_discrete mutual_info_with_entropy, mutual_info_score, mutual_info_classif]

current_question: question_2
```

## Reports

1. After excuting the pipeline, all generated figures would be saved in <strong><em>[reports/figures](reports/figures)</em></strong>.

2. The details analysis and summary of results of question 1 is stored in [Exploratory_Data_Analysis_Report.docx](reports/figures/Exploratory_Data_Analysis_Report.docx).

3. The results of question 2 will be displayed directly on the terminal when being executed.

4. All of the initial analysis and codes can be found within [notebooks](notebooks)
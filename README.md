# Predicting NBA Draft Positions of NCAA Basketball Players using Neural Networks

# Introduction
In my other repository, I scraped data of NCAA Basketball Players between the year 2005-2019. In this repository, I will predict how players will fair in the draft.

Initially, I had 2 main tasks in mind, which was to predict the draft position by category (Top 15 pick, 16th-30th pick, 31st-45th pick, 46th to 60th pick, or Undrafted), and also to predict the draft position of players (1 to 60) by regressing th features on the position. However, both tasks proved to be too difficult for the model. When I tried predicting whether a NCAA Basketball player will be drafted or undrafted, the results proved to be much better.

The reasons why the two initial task proved to be too difficult is largely due to two factors. First, there is a lack of data. There are only ~100 rows of data per year, and I only used data from 2005-2019 as many advanced statistics were not recorded before then. Secondly, draft position is largely dependent on potential of players as well, and potential is difficult to represent well in statistics.

Nevertheless, I started on this project despite knowing that it will likely not produce good results, mainly to practice web scraping using BeautifulSoup and deep learning using the PyTorch framework.

# Getting Started

## Prerequisites
Python 3.8, `conda` and `pip`.

## Installation
```
conda install jupyter
conda install -c anaconda numpy pandas
conda install -c pytorch pytorch
conda install -c conda-forge beautifulsoup4 requests
pip install sklearn
conda install -c anaconda matplotlib
pip install argparse
```

# Running the workflow

## Data Cleaning
The cleaning of the scraped data is done in ```cleaning.ipynb```

## Neural Networks
The model implemented in PyTorch trains on the train dataset, which has a 0.8/0.1/0.1 train/val/test split.
```
python main.py classification1
python main.py classification2
python main.py regression
```
*'classification1' refers to the first task (predicting into 5 categories) and 'classification2' refers to predicting draft/undraft.*

The above code trains and test the model, outputting the accuracy/MAE. The parameters and layers are a result of hyperparameters tuning done in ```experiments.ipynb```. These parameters and layers can be adjusted in ```main.py``` as well.

# Author
Chua Jing Yang

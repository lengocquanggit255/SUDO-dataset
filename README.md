## Overview
This repository contains implementations and data. The proposed abstractive summarization model shown in the following Figure

<img width="1121" height="707" alt="model_baselines_05012026" src="https://github.com/user-attachments/assets/6a2e7cff-2682-4b6f-b1f4-9192c49abcb4" />

## Structure of the Repository
The structure of the repository is presented as:
- `data`: the folder contains the dataset in csv format, along with the guidelines used during the data labeling process.
- `code`: the folder contain the source code of the project, include the notebooks for training and inference with each component of the model.

## Data Schema

### 1. Reviews Data (`data/reviews/beer-com-reviews_*.csv`)
This dataset contains comparative reviews between two different beers by the same user.
- `profileName`: The account name of the reviewer.
- **Beer 1 Features (Suffix `_1`):**
  - `beerName_1`: Name of the first beer.
  - `beerId_1`: Unique identifier for the first beer.
  - `beerABV_1`: Alcohol By Volume (ABV) of the first beer.
  - `beerStyle_1`: Style/category of the first beer.
  - `reviewText_1`: The full text review written by the user for the first beer.
  - `appearanceRate_1`: Rating for appearance.
  - `aromaRate_1`: Rating for aroma.
  - `palateRate_1`: Rating for palate/mouthfeel.
  - `tasteRate_1`: Rating for taste.
- **Beer 2 Features (Suffix `_2`):**
  - `beerName_2`: Name of the first beer.
  - `beerId_2`: Unique identifier for the first beer.
  - `beerABV_2`: Alcohol By Volume (ABV) of the first beer.
  - `beerStyle_2`: Style/category of the first beer.
  - `reviewText_2`: The full text review written by the user for the first beer.
  - `appearanceRate_2`: Rating for appearance.
  - `aromaRate_2`: Rating for aroma.
  - `palateRate_2`: Rating for palate/mouthfeel.
  - `tasteRate_2`: Rating for taste.
- **Comparison Labels:**
  - `appearance`: Categorical labels, (0 - equal, 1 - reviewText_1 better than reviewText_2, -1 - reviewText_1 worse than reviewText_2, null - cannot compare).
  - `aroma`: Categorical labels, (0 - equal, 1 - reviewText_1 better than reviewText_2, -1 - reviewText_1 worse than reviewText_2, null - cannot compare).
  - `palate`: Categorical labels, (0 - equal, 1 - reviewText_1 better than reviewText_2, -1 - reviewText_1 worse than reviewText_2, null - cannot compare).
  - `taste`: Categorical labels, (0 - equal, 1 - reviewText_1 better than reviewText_2, -1 - reviewText_1 worse than reviewText_2, null - cannot compare)


### 2. Sentences Data (`data/sentences/beer-com-sentences_*.csv`)
This dataset breaks down the reviews into individual sentences and includes specific aspect sentiment labels.
- `profileName`: The account name of the reviewer.
- `beerId`: Unique identifier for the beer.
- `beerName`: Name of the beer.
- `beerABV`: Alcohol By Volume (ABV).
- `beerStyle`: Style/category of the beer.
- `reviewSentence`: A single sentence extracted from the full review.
- `appearanceRate`, `aromaRate`, `palateRate`, `tasteRate`: The overall aspect ratings given by the user for this beer.
- `appearance`, `aroma`, `palate`, `taste`: Binary labels indicating whether the current `reviewSentence` discusses these specific aspects (e.g., 1 if the sentence talks about aroma, 0 otherwise).

## Dependencies requirements:
All the requirements are listed in file `requirements.txt`, simply install it by run the command:
```bash
pip install -r code/requirements.txt
```

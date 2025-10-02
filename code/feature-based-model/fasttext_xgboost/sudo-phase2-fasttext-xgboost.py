# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# ============================
# 1. Setup: download FastText
# ============================

# %%
import fasttext
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
from datasets import load_dataset
import itertools
from tqdm import tqdm
import json

# -------------------
# 1. Load pretrained fastText
# -------------------
print("Loading fastText model...")
model = fasttext.load_model("src/models/cv_assessment/match_cv/model/cc.en.300.bin")

def get_embedding(text):
    return model.get_sentence_vector(str(text))

# Map nhãn {-1,0,1} -> {0,1,2}
label2id = {-1: 0, 0: 1, 1: 2}
id2label = {v: k for k, v in label2id.items()}

def build_features(split_ds, aspect):
    X, y = [], []
    for row in tqdm(split_ds, desc=f"Building features for {aspect}"):
        v1 = get_embedding(row["sentences_1"])
        v2 = get_embedding(row["sentences_2"])
        vec = np.concatenate([v1, v2])
        X.append(vec)
        y.append(label2id[int(row[aspect])])   # sử dụng aspect thay vì "appearance"
    return np.array(X), np.array(y)

# -------------------
# 2. Train models for each aspect
# -------------------
aspects = ["appearance", "aroma", "palate", "taste"]
results = {}

param_grid = {
    "max_depth": [3, 5, 7],           
    "eta": [0.05, 0.1],          
    "subsample": [0.8, 1.0],           
    "colsample_bytree": [0.8, 1.0],    
    "min_child_weight": [1, 3, 5] ,                
    "num_boost_round": [5000]
}

for aspect in aspects:
    print(f"\n{'='*50}")
    print(f"Training model for aspect: {aspect}")
    print(f"{'='*50}")
    
    # Load dataset for current aspect
    dataset = load_dataset(f"trungpq/rlcc-new-data-{aspect}")
    
    # drop null ở sentences_1, sentences_2, aspect
    def not_null(example):
        return (
            example["sentences_1"] is not None 
            and example["sentences_2"] is not None 
            and example[aspect] is not None
        )
    
    dataset = dataset.filter(not_null)
    
    X_train, y_train = build_features(dataset["train"], aspect)
    X_valid, y_valid = build_features(dataset["validation"], aspect)
    X_test,  y_test  = build_features(dataset["test"], aspect)
    
    # -------------------
    # 3. Hyperparameter search trên validation (multi-class)
    # -------------------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    num_classes = len(label2id)
    
    best_params = None
    best_f1 = -1
    
    print(f"Searching best hyperparameters for {aspect}...")
    param_combinations = list(itertools.product(*param_grid.values()))
    for values in tqdm(param_combinations, desc=f"Hyperparameter search - {aspect}"):
        params = dict(zip(param_grid.keys(), values))
        
        # Extract num_boost_round from params
        num_boost_round = params.pop("num_boost_round")
        
        params.update({
            "objective": "multi:softmax",   
            "eval_metric": "mlogloss",
            "num_class": num_classes,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1
        })
    
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=500,
            verbose_eval=100
        )
    
        y_pred_val = bst.predict(dvalid)   
        f1 = f1_score(y_valid, y_pred_val, average="macro")  
    
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            print(f"New best F1={f1:.4f} for {aspect} with params={params}")
    
    print(f"\nBest Params for {aspect}:", best_params)
    print(f"Best Validation F1 for {aspect}:", best_f1)
    
    # -------------------
    # 4. Train lại với train+validation
    # -------------------
    print(f"\nRetraining {aspect} model with best params on train+validation...")
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.concatenate([y_train, y_valid])
    
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    bst = xgb.train(
        params=best_params,
        dtrain=dtrain_full,
        num_boost_round=2000,
        evals=[(dtest, "test")],
        early_stopping_rounds=250,
        verbose_eval=50
    )
    
    # -------------------
    # 5. Đánh giá cuối trên test set
    # -------------------
    y_pred_ids = bst.predict(dtest)  
    y_pred = np.array([id2label[int(i)] for i in y_pred_ids])  
    y_test_true = np.array([id2label[int(i)] for i in y_test]) 
    
    acc = accuracy_score(y_test_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_true, y_pred, average="macro")
    
    # Store results
    results[aspect] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "best_params": best_params,
        "model": bst
    }
    
    print(f"\nFinal Test Evaluation for {aspect}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    print(f"\nClassification Report for {aspect}:")
    print(classification_report(y_test_true, y_pred, digits=4))

# -------------------
# 6. Summary of all results
# -------------------
print(f"\n{'='*60}")
print("SUMMARY OF ALL ASPECTS")
print(f"{'='*60}")

for aspect in aspects:
    result = results[aspect]
    print(f"\n{aspect.upper()}:")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1-score:  {result['f1_score']:.4f}")

# -------------------
# 7. Save best hyperparameters to JSON
# -------------------
best_hyperparams = {}
for aspect in aspects:
    best_hyperparams[aspect] = {
        "best_params": results[aspect]["best_params"],
        "f1_score": results[aspect]["f1_score"],
        "accuracy": results[aspect]["accuracy"],
        "precision": results[aspect]["precision"],
        "recall": results[aspect]["recall"]
    }

# Save to JSON file
output_file = "best_hyperparams_xgboost_fasttext.json"
with open(output_file, 'w') as f:
    json.dump(best_hyperparams, f, indent=4)

print(f"\nBest hyperparameters saved to: {output_file}")

# %%




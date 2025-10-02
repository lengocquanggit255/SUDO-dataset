import numpy as np
import pandas as pd
import fasttext
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
from datasets import load_dataset
from tqdm import tqdm
import json
import os

# Create checkpoint directory
os.makedirs("temp/fasttext-xgboost/xgboost_checkpoint", exist_ok=True)

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------
# 1. Load pretrained fastText
# -------------------
print("Loading fastText model...")
model = fasttext.load_model("temp/fasttext_xgboost/fasttext_checkpoint/cc.en.300.bin")

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
        y.append(label2id[int(row[aspect])])
    return np.array(X), np.array(y)

# -------------------
# 2. Best hyperparameters for each aspect
# -------------------
best_hyperparams = {
    "appearance": {
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1
    },
    "aroma": {
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1
    },
    "palate": {
        'max_depth': 7,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'min_child_weight': 5,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1
    },
    "taste": {
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1
    }
}

# -------------------
# 3. Train models for each aspect with best params
# -------------------
aspects = ["appearance", "aroma", "palate", "taste"]
results = {}

for aspect in aspects:
    print(f"\n{'='*50}")
    print(f"Training {aspect} model with best hyperparameters")
    print(f"{'='*50}")
    
    # Load dataset for current aspect
    dataset = load_dataset(f"trungpq/rlcc-new-data-{aspect}")
    
    # Filter null values
    def not_null(example):
        return (
            example["sentences_1"] is not None 
            and example["sentences_2"] is not None 
            and example[aspect] is not None
        )
    
    dataset = dataset.filter(not_null)
    
    # Build features
    X_train, y_train = build_features(dataset["train"], aspect)
    X_valid, y_valid = build_features(dataset["validation"], aspect)
    X_test, y_test = build_features(dataset["test"], aspect)
    
    # Combine train and validation for final training
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.concatenate([y_train, y_valid])
    
    # Create DMatrix
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Get best parameters for this aspect
    params = best_hyperparams[aspect].copy()
    
    print(f"Using parameters: {params}")
    
    # Train model with best parameters
    print(f"Training final model for {aspect}...")
    bst = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=2000,
        evals=[(dtest, "test")],
        early_stopping_rounds=250,
        verbose_eval=50
    )
    
    # Evaluate on test set
    y_pred_ids = bst.predict(dtest)
    y_pred = np.array([id2label[int(i)] for i in y_pred_ids])
    y_test_true = np.array([id2label[int(i)] for i in y_test])
    
    # Calculate metrics
    acc = accuracy_score(y_test_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_true, y_pred, average="macro")
    
    # Store results
    results[aspect] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "best_params": params,
        "model": bst
    }
    
    # Save model
    model_path = f"temp/fasttext-xgboost/xgboost_checkpoint/{aspect}_best_model.json"
    bst.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    print(f"\nFinal Test Evaluation for {aspect}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    print(f"\nClassification Report for {aspect}:")
    print(classification_report(y_test_true, y_pred, digits=4))

# -------------------
# 4. Summary of all results
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
# 5. Save final results to JSON
# -------------------
final_results = {}
for aspect in aspects:
    final_results[aspect] = {
        "best_params": results[aspect]["best_params"],
        "f1_score": float(results[aspect]["f1_score"]),
        "accuracy": float(results[aspect]["accuracy"]),
        "precision": float(results[aspect]["precision"]),
        "recall": float(results[aspect]["recall"])
    }

# Save results
output_file = "temp/fasttext-xgboost/final_results_xgboost_fasttext.json"
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=4)

print(f"\nFinal results saved to: {output_file}")
print("All models saved successfully!")

import pandas as pd
import numpy as np
import fasttext
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from datasets import load_dataset
from tqdm import tqdm
import json
import pickle
import os

# -----------------
# 1. Load FastText model
# -----------------
print("Loading fastText model...")
fasttext_model = fasttext.load_model("temp/fasttext-svm/fasttext-checkpoint/cc.en.300.bin")

def get_embedding(text):
    return fasttext_model.get_sentence_vector(str(text))

def build_features(df):
    """Build features by concatenating FastText embeddings of text1 and text2"""
    X = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
        v1 = get_embedding(row["text1"])
        v2 = get_embedding(row["text2"])
        vec = np.concatenate([v1, v2])
        X.append(vec)
    return np.array(X)

# -----------------
# 2. Setup aspects and results tracking
# -----------------
aspects = ["appearance", "aroma", "palate", "taste"]
results = {}

# -----------------
# 3. Updated Grid Search parameters for SVM
# -----------------
param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto", 0.01, 0.001]
}

# -----------------
# 4. Train models for each aspect
# -----------------
for aspect in aspects:
    print(f"\n{'='*50}")
    print(f"Training model for aspect: {aspect}")
    print(f"{'='*50}")
    
    # Load dataset for current aspect
    dataset = load_dataset(f"trungpq/rlcc-new-data-{aspect}")

    df_train = dataset["train"].to_pandas()
    df_val   = dataset["validation"].to_pandas()
    df_test  = dataset["test"].to_pandas()

    # Chuẩn hóa cột
    for df in [df_train, df_val, df_test]:
        df.rename(columns={
            "sentences_1": "text1",
            "sentences_2": "text2",
            aspect: "label"  # Use dynamic aspect name
        }, inplace=True)
        df.dropna(subset=["text1", "text2", "label"], inplace=True)

    # Build features using FastText embeddings
    print(f"Building FastText features for {aspect}...")
    X_train = build_features(df_train)
    X_val = build_features(df_val)
    X_test = build_features(df_test)
    
    y_train = df_train["label"].values
    y_val = df_val["label"].values
    y_test = df_test["label"].values

    # SVM model
    svm_model = SVC(class_weight="balanced", random_state=42)

    # Grid Search trên train+val
    print(f"Performing grid search for {aspect}...")
    
    # concat train + val (để cross-validation nội bộ)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    grid_search = GridSearchCV(
        svm_model,
        param_grid,
        scoring="f1_macro",
        cv=5,             # cross-validation thật sự trong train+val
        n_jobs=-1,
        verbose=2
    )

    # Wrap grid search with tqdm
    with tqdm(total=1, desc=f"Grid search for {aspect}") as pbar:
        grid_search.fit(X_trainval, y_trainval)
        pbar.update(1)

    print(f"Best Params for {aspect}:", grid_search.best_params_)
    print(f"Best CV Score for {aspect}:", grid_search.best_score_)

    # Đánh giá trên test
    best_model = grid_search.best_estimator_

    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro")
    
    # Store results
    results[aspect] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "model": best_model
    }
    
    print(f"\nFinal Test Evaluation for {aspect}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    print(f"\nClassification Report for {aspect}:")
    print(classification_report(y_test, y_test_pred, digits=4))

# -----------------
# 4. Summary of all results
# -----------------
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

# -----------------
# 5. Save best hyperparameters to JSON
# -----------------
best_hyperparams = {}
for aspect in aspects:
    # Convert numpy types to native Python types for JSON serialization
    best_hyperparams[aspect] = {
        "best_params": results[aspect]["best_params"],
        "best_cv_score": float(results[aspect]["best_cv_score"]),
        "f1_score": float(results[aspect]["f1_score"]),
        "accuracy": float(results[aspect]["accuracy"]),
        "precision": float(results[aspect]["precision"]),
        "recall": float(results[aspect]["recall"])
    }

# Save to JSON file
output_file = "best_hyperparams_tfidf_svm.json"
with open(output_file, 'w') as f:
    json.dump(best_hyperparams, f, indent=4)

print(f"\nBest hyperparameters saved to: {output_file}")

# -----------------
# 6. Save trained models
# -----------------
# Create directory if it doesn't exist
os.makedirs("svm_checkpoint", exist_ok=True)

for aspect in aspects:
    model_path = f"svm_checkpoint/{aspect}_best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(results[aspect]["model"], f)
    print(f"Saved {aspect} model to: {model_path}")

print("\nAll models saved successfully!")

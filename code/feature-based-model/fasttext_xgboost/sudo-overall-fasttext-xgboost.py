# %%
# !nvidia-smi
# !pip show scikit-learn
# !pip install scikit-learn==1.5.1
# !pip install fasttext-wheel
# !pip install xgboost

# %%
import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PretrainedConfig, PreTrainedModel, BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import json
import fasttext
import numpy as np
import xgboost as xgb

import warnings

# Bỏ qua tất cả cảnh báo UserWarning (trong đó có InconsistentVersionWarning cũ)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


nltk.download('punkt')
nltk.download('punkt_tab')

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
class SLACBERTModelConfig(PretrainedConfig):
    model_type = "bert_model"

    def __init__(self, num_classes=1, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.pos_weight = pos_weight 

class SLACBERTModel(PreTrainedModel):
    config_class = SLACBERTModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, config.num_classes)

        if config.pos_weight is not None:
            pos_weight = torch.tensor(config.pos_weight, dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        

    def forward(self, input_ids=None, attention_mask=None, labels=None):      
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

# %%
class XGBoostComparativeModel:
    def __init__(self, aspect):
        # Load the pretrained FastText model for embeddings
        self.fasttext_model = fasttext.load_model("temp/fasttext_xgboost/fasttext_checkpoint/cc.en.300.bin")
        
        # Load the trained XGBoost model for this aspect
        self.aspect = aspect
        self.xgb_model = xgb.Booster()
        
        # Load the trained model from JSON file
        model_path = f"temp/fasttext_xgboost/xgboost_checkpoint/{aspect}_best_model.json"
        self.xgb_model.load_model(model_path)
        
        # Label mapping
        self.label2id = {-1: 0, 0: 1, 1: 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def get_embedding(self, text):
        """Get FastText embedding for text"""
        return self.fasttext_model.get_sentence_vector(str(text))
    
    def predict(self, text_1, text_2):
        """Predict comparison between two texts"""
        # Get embeddings for both texts
        emb_1 = self.get_embedding(text_1)
        emb_2 = self.get_embedding(text_2)
        
        # Concatenate embeddings as features
        features = np.concatenate([emb_1, emb_2]).reshape(1, -1)
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(features)
        
        # Get prediction (returns class id)
        pred_id = int(self.xgb_model.predict(dtest)[0])
        
        # Convert back to original label
        return self.id2label[pred_id]

# %%
class OverallMode(nn.Module):
    def __init__(self, aspect, slac_id, device):
        super().__init__()
        self.aspect = aspect
        self.device = device
        
        self.aspect_index = {'appearance': 0, 'aroma': 1, 'palate': 2, 'taste': 3}
        self.aspect_idx = self.aspect_index[aspect]  
        
        # Load SLAC model for aspect sentence classification
        self.slac_model = SLACBERTModel.from_pretrained(slac_id).to(device)
        self.slac_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load XGBoost model for comparative classification
        self.xgb_model = XGBoostComparativeModel(aspect)
            
    @staticmethod
    def _truncate_seq(tokens, max_length):
        while True:
            total_length = len(tokens)
            if total_length <= max_length:
                break
            tokens.pop()
            
        return tokens
    
    @staticmethod
    def _split_clean_sentences(text):
        sentences = sent_tokenize(text.lower()) 
        sentences = [re.sub(r'\W+', ' ', s).strip() for s in sentences if len(word_tokenize(s)) > 1]  
        return sentences
    
    def _get_aspect_sentences(self, review_sentences):
        if len(review_sentences) == 0:
            return []
            
        total_input_ids = []
        total_input_mask = []
        
        for sent in review_sentences:
            tokens = self.tokenizer.tokenize(sent)
            self._truncate_seq(tokens, 128 - 2) #account for [CLS] [SEP]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (128 - len(input_ids))
            
            input_ids += padding
            input_mask += padding
            
            total_input_ids.append(torch.tensor([input_ids], dtype=torch.long).to(self.device))
            total_input_mask.append(torch.tensor([input_mask], dtype=torch.long).to(self.device))
            
        input_ids = torch.cat(total_input_ids, dim=0).to(self.device)
        attention_masks = torch.cat(total_input_mask, dim=0).to(self.device)
        
        with torch.no_grad():
            aspect_logits = self.slac_model(input_ids, attention_masks).logits
            probs = torch.sigmoid(aspect_logits.clone().detach())
            
            aspect_sentences = []
            for prob, sent in zip(probs, review_sentences):
                if prob > 0.5:
                    aspect_sentences.append(sent)
                
            return aspect_sentences
        
    def forward(self, review_1, review_2):
        review_1_sent = self._split_clean_sentences(review_1)
        review_2_sent = self._split_clean_sentences(review_2)
        
        if len(review_1_sent) == 0 or len(review_2_sent) == 0:
            return 2
        
        review_1_aspect_sentences = self._get_aspect_sentences(review_1_sent)
        review_2_aspect_sentences = self._get_aspect_sentences(review_2_sent)
        
        if len(review_1_aspect_sentences) == 0 or len(review_2_aspect_sentences) == 0:
            return 2

        # Combine aspect sentences into single texts
        review_1_text = " ".join(review_1_aspect_sentences)
        review_2_text = " ".join(review_2_aspect_sentences)
        
        # Use XGBoost model for comparison
        try:
            result = self.xgb_model.predict(review_1_text, review_2_text)
            return result
        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            # Fallback to no comparison if XGBoost fails
            return 2

# %%
# Define labels
labels = [-1, 0, 1, 2]
positive_labels = [-1, 0, 1]  # Positive labels: A<B, A=B, A>B
negative_label = 2            # Negative label: No comparison

from collections import defaultdict
def compute_metrics(true_labels, pred_labels):
        # Initialize dictionaries to count
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    # Process each sample
    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            # True Positive: Only count for positive labels
            if true in positive_labels:
                tp[true] += 1
        else:
            # True is positive, Prediction is null (missing)
            if true in positive_labels and pred == negative_label:
                fn[true] += 1  # Only increase FN
            
            # True is null, Prediction is positive (excess)
            elif true == negative_label and pred in positive_labels:
                fp[pred] += 1  # Only increase FP
            
            # Confusion between positive labels
            elif true in positive_labels and pred in positive_labels:
                fn[true] += 1  # Missing the correct label
                fp[pred] += 1  # Excess of the predicted label
    
    return tp, fp, fn

# %%
from tqdm import tqdm

# Define a function to run evaluation for all aspects
def overall_evaluate_all_aspects(sampling_method, eval_dataset):
    aspects = ['appearance', 'aroma', 'palate', 'taste']
    
    # Dictionary to store predictions for each aspect
    aspect_predictions = {}
    
    # Run each aspect model and collect predictions
    for aspect in aspects:
        model = OverallMode(
            aspect, 
            f"trungpq/slac-new-{aspect}-{sampling_method}",  
            device
        )
        
        predictions = []
        for sample in tqdm(eval_dataset):
            pred = model(sample['reviewText_1'], sample['reviewText_2'])
            predictions.append(pred)
        
        # Store predictions for this aspect
        aspect_predictions[aspect] = predictions
        del model
    
    # Combine all predictions and true labels into flat lists
    all_predictions_flat = []
    all_true_labels_flat = []
    
    for i in range(len(eval_dataset)):
        for aspect in aspects:
            all_predictions_flat.append(aspect_predictions[aspect][i])
            all_true_labels_flat.append(eval_dataset[i][aspect])
    
    # Calculate TP, FP, FN
    tp, fp, fn = compute_metrics(all_true_labels_flat, all_predictions_flat)
    
    # Calculate precision, recall, F1 for each class
    class_metrics = {}
    for label in positive_labels:
        # Skip if no instances for this class
        if tp[label] + fp[label] == 0:
            precision = 0
        else:
            precision = tp[label] / (tp[label] + fp[label])
            
        if tp[label] + fn[label] == 0:
            recall = 0
        else:
            recall = tp[label] / (tp[label] + fn[label])
            
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
            
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp[label],
            'fp': fp[label],
            'fn': fn[label]
        }
    
    # Calculate macro metrics (average over classes)
    macro_precision = sum(m['precision'] for m in class_metrics.values()) / len(positive_labels)
    macro_recall = sum(m['recall'] for m in class_metrics.values()) / len(positive_labels)
    macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(positive_labels)
    
    # Calculate micro metrics (aggregate TP, FP, FN)
    total_tp = sum(tp[label] for label in positive_labels)
    total_fp = sum(fp[label] for label in positive_labels)
    total_fn = sum(fn[label] for label in positive_labels)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Return the combined results
    return {
        'method': f"{sampling_method}_xgboost",
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall
    }

# %%
# Define a function to evaluate each aspect separately
def aspect_wise_evaluate(sampling_method, eval_dataset):
    aspects = ['appearance', 'aroma', 'palate', 'taste']
    aspect_results = {}
    
    for aspect in aspects:
        print(f"Evaluating {aspect} model for aspect-wise metrics...")
        model = OverallMode(
            aspect, 
            f"trungpq/slac-new-{aspect}-{sampling_method}", 
            device
        )
        
        predictions = []
        true_labels = []
        
        for i, sample in enumerate(tqdm(eval_dataset)):
            pred = model(sample['reviewText_1'], sample['reviewText_2'])
            predictions.append(pred)
            true_labels.append(sample[aspect])
        
        # Calculate TP, FP, FN for this aspect
        tp, fp, fn = compute_metrics(true_labels, predictions)
        
        # Calculate precision, recall, F1 for each class
        class_metrics = {}
        for label in positive_labels:
            # Skip if no instances for this class
            if tp[label] + fp[label] == 0:
                precision = 0
            else:
                precision = tp[label] / (tp[label] + fp[label])
                
            if tp[label] + fn[label] == 0:
                recall = 0
            else:
                recall = tp[label] / (tp[label] + fn[label])
                
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
                
            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp[label],
                'fp': fp[label],
                'fn': fn[label]
            }
        
        # Calculate macro metrics (average over classes)
        macro_precision = sum(m['precision'] for m in class_metrics.values()) / len(positive_labels)
        macro_recall = sum(m['recall'] for m in class_metrics.values()) / len(positive_labels)
        macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(positive_labels)
        
        # Calculate micro metrics (aggregate TP, FP, FN)
        total_tp = sum(tp[label] for label in positive_labels)
        total_fp = sum(fp[label] for label in positive_labels)
        total_fn = sum(fn[label] for label in positive_labels)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        aspect_results[aspect] = {
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics
        }
        
        print(f"{aspect} - Micro P/R/F1: {micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}")
        print(f"{aspect} - Macro P/R/F1: {macro_precision:.4f}/{macro_recall:.4f}/{macro_f1:.4f}")
        
        del model
    
    return aspect_results

# %%
import datasets

eval_dataset = datasets.load_dataset("lengocquangLAB/beer-com-reviews", split="test")

for sampling_method in ['upsample_replacement', "class_weight"]:
    print("-"*50)
    print(f"Evaluating with sampling method: {sampling_method}")
    results = overall_evaluate_all_aspects(sampling_method, eval_dataset)
    print(results)
    print()

# %%
# Run aspect-wise evaluation
print("="*70)
print("ASPECT-WISE EVALUATION")
print("="*70)

for sampling_method in ['upsample_replacement', "class_weight"]:
    print("-"*50)
    print(f"Aspect-wise evaluation with sampling method: {sampling_method}")
    aspect_results = aspect_wise_evaluate(sampling_method, eval_dataset)
    
    # Print summary for each aspect
    for aspect, metrics in aspect_results.items():
        print(f"\n{aspect.upper()} Results:")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print()



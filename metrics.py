from collections import Counter
import re
import string
# import evaluate
# import torch
# from transformers import AutoTokenizer, AutoModel
import numpy as np

def normalize_answer(s: str) -> str:
    
    def lower(text: str) -> str:
        return text.lower()

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0, "precision": 0, "recall": 0}
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}

# bleu = evaluate.load("bleu")
# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")

# # Load RoBERTa-Large model and tokenizer
# _roberta_model = None
# _roberta_tokenizer = None

# def _get_roberta_model():
#     global _roberta_model, _roberta_tokenizer
#     if _roberta_model is None:
#         model_name = "FacebookAI/roberta-large"
#         _roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
#         _roberta_model = AutoModel.from_pretrained(model_name)
#         _roberta_model.eval()
#     return _roberta_model, _roberta_tokenizer

# def bleu_reward(prediction, ground_truth):
#     # Handle empty references to prevent ZeroDivisionError
#     if not ground_truth or not ground_truth.strip():
#         return 0.0
#     try:
#         bleu_score = bleu.compute(predictions=[prediction], references=[ground_truth], smooth=True)
#         brevity_penalty = bleu_score["brevity_penalty"]
#         if brevity_penalty == 0:
#             return 0.0
#         return bleu_score["bleu"] / brevity_penalty
#     except (ZeroDivisionError, ValueError):
#         return 0.0
#     #return bleu_score["bleu"], bleu_score["brevity_penalty"]

# # def bleu_no_bp(prediction, ground_truth):
# #     bleu_score = bleu.compute(predictions=[prediction], references=[ground_truth], smooth=True)
# #     return bleu_score["bleu"] / bleu_score["brevity_penalty"]

# # def bleu_bp_only(prediction, ground_truth):
# #     bleu_score = bleu.compute(predictions=[prediction], references=[ground_truth], smooth=True)
# #     return bleu_score["brevity_penalty"]

# def rouge_reward(prediction, ground_truth):
#     rouge_score = rouge.compute(predictions=[prediction], references=[ground_truth])
#     return rouge_score["rougeL"]

# def bleu_rouge_f1_reward(prediction, ground_truth):
#     # Handle empty references to prevent ZeroDivisionError
#     if not ground_truth or not ground_truth.strip():
#         return 0.0
#     try:
#         bleu_score = bleu.compute(predictions=[prediction], references=[ground_truth], smooth=True)
#         rouge_score = rouge.compute(predictions=[prediction], references=[ground_truth])
#         return (bleu_score["bleu"] * rouge_score["rougeL"] * 2) / (bleu_score["bleu"] + rouge_score["rougeL"] + 1e-10)
#     except (ZeroDivisionError, ValueError):
#         return 0.0

# def bertscore_reward(prediction, ground_truth):
#     bertscore_score = bertscore.compute(predictions=[prediction], references=[ground_truth], model_type="distilbert-base-uncased")
#     return bertscore_score["f1"][0]

# def roberta_embedding_similarity(prediction, ground_truth):
#     """
#     Compute cosine similarity between prediction and ground_truth using RoBERTa-Large embeddings.
    
#     Args:
#         prediction: The predicted text string
#         ground_truth: The ground truth text string
    
#     Returns:
#         float: Cosine similarity score between 0 and 1
#     """
#     model, tokenizer = _get_roberta_model()
    
#     # Tokenize and encode both texts
#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # Use mean pooling of the last hidden state
#             embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
#         return embedding.numpy()
    
#     pred_embedding = get_embedding(prediction)
#     gt_embedding = get_embedding(ground_truth)
    
#     # Compute cosine similarity
#     similarity = np.dot(pred_embedding, gt_embedding) / (np.linalg.norm(pred_embedding) * np.linalg.norm(gt_embedding))
#     return float(similarity)
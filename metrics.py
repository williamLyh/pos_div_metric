from typing import List
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datasets import Dataset
from torch.nn.functional import kl_div
from scipy.stats import pearsonr


def predict_sentence_discourse(data, batch_size=16):
    """
    Predicts the discourse sequence for a given set of sentences.

    Args:
        data (Dataset): The dataset containing the sentences.
        batch_size (int): The batch size to use for prediction.

    Returns:
        list: The predicted discourse sequence.
    """
    classifier_model_name = '/home/yinhong/Documents/source/InstructDiscourse/model-checkpoint/news-discourse-classifier/checkpoint-3260'
    # classifier_model_name = '/mnt/nas_home/yl535/InstructDiscourse/model-checkpoint/news-discourse-classifier/checkpoint-3260'
    bert_tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_model_name)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    # Convert data from list of string to Dataset
    if isinstance(data, list):
        data = Dataset.from_dict({'input': data})

    train_loader = DataLoader(data, batch_size=batch_size)
    labels_pred = []

    for batch in train_loader:
        batch_input = bert_tokenizer(
            batch['input'],
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        outputs = model(
            batch_input['input_ids'].to(device),
            attention_mask=batch_input['attention_mask'].to(device)
        )

        class_score = torch.softmax(outputs.logits, dim=-1)
        class_prediction = torch.argmax(class_score, dim=-1)
        labels_pred += class_prediction.cpu().tolist()
    return labels_pred


def calculate_discourse_accuracy(predict_seq, reference_seq):
    """
    Calculates the accuracy of the predicted discourse sequence compared to the reference sequence.

    Args:
        predict_seq (list): The predicted discourse sequence.
        reference_seq (list): The reference discourse sequence.

    Returns:
        float: The accuracy of the predicted discourse sequence compared to the reference sequence.
    """
    accuracy = []
    for predicted_plan, reference_plan in zip(predict_seq, reference_seq):
        accuracy.append(exact_match(predicted_plan, reference_plan))
    return np.average(accuracy)


def exact_match(preds, refs):
    """
    Calculates the exact match between two sequences.

    Args:
        preds (list): The predicted sequence.
        refs (list): The reference sequence.

    Returns:
        float: The exact match between the two sequences.
    """
    matched_cnt = 0
    for i, ref in enumerate(refs):
        if i < len(preds) and ref == preds[i]:
            matched_cnt += 1
    return min(matched_cnt / len(refs), 1)


def calculate_positional_distribution(label_doc_list, num_class=8, num_bins=10):
    """
    Calculates the positional distribution of the document.
    The positional distribution is the distribution of the labels at certain position of documents.
    The bin is defined by the split_bins function.

    Args:
        label_doc_list (list): The list of labels for each document.
        num_bins_default (int): The number of bins to use for the distribution.

    Returns:
        list: The positional label density.
    """
    positional_label_distribution = np.zeros((num_bins, num_class))
    for stage_label_doc in label_doc_list:
        bin_cnt, bin_values = np.histogram(range(len(stage_label_doc)), bins=num_bins)
        for idx, label in enumerate(stage_label_doc):
            bin_idx = np.digitize(idx, bin_values[:-1])-1
            positional_label_distribution[bin_idx][label] += 1
    # normalize across class
    positional_label_density = positional_label_distribution / positional_label_distribution.sum(axis=1, keepdims=True)
    return positional_label_density


def calculate_positional_divergence(predictions: List[List[int]], references: List[List[int]], num_class=8, num_bins_default=10, return_normalized_score=False, return_tensor=True) -> float:
    # get the suitable bin size. 
    # Firstly get maximum bin size that can be spanned by the reference and response discourse (with no empty bin).
    # Then get the minimum of the maximum bin size and the default bin size.
    max_pred_discourse_len = max([len(discourse) for discourse in predictions])
    max_ref_discourse_len = max([len(discourse) for discourse in references])
    num_bins = min(num_bins_default, max_pred_discourse_len, max_ref_discourse_len)

    generated_positional_distribution = calculate_positional_distribution(predictions, num_class=num_class, num_bins=num_bins)
    reference_positional_distribution = calculate_positional_distribution(references, num_class=num_class, num_bins=num_bins)

    pred_log_prob = torch.log(torch.tensor(generated_positional_distribution)+1e-8)   # Convert to log probability with epsilon
    ref_log_prob = torch.log(torch.tensor(reference_positional_distribution)+1e-8)   # Convert to log probability with epsilon

    pos_div = kl_div(
        input=pred_log_prob,
        target=ref_log_prob, 
        reduction='batchmean', 
        log_target=True
    )
    if not return_tensor:
        pos_div = pos_div.item()
        
    if return_normalized_score:
        normalized_pos_div = 1/(1+pos_div)
        return normalized_pos_div
    return pos_div


def sentence_splitter(text_doc, flat=False):
    """
    Splits a document into sentences.
    Args:
        text_doc (list): A list of strings representing the document.
        flat (bool, optional): If True, the sentences will be flattened into a single list. 
                               If False, the sentences will be stored as a list of lists.
    Returns:
        list: A list of sentences. If flat is True, the sentences are flattened into a single list.
              If flat is False, the sentences are stored as a list of lists.
    """
    splitted_doc = []
    for doc in text_doc:
        ans = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!|\’|\”)(?<![0-9]\.)\s(?!\w\.)(?=[A-Z])', doc)
        if flat:
            splitted_doc += ans
        else:
            splitted_doc.append(ans)
    return splitted_doc


def evaluate_corr(pred_discourse, ref_discourse, human_scores,  num_class=8, num_bins_default=5):
    acc_list = []
    pos_div_list = []
    for pred, ref in zip(pred_discourse, ref_discourse):
        acc_list.append(exact_match(preds=pred, refs=ref))
        pos_div_list.append(calculate_positional_divergence(
                                predictions=[pred], 
                                references=[ref], 
                                num_class=num_class, 
                                num_bins_default=num_bins_default, 
                                return_normalized_score=True
                                )
                            )
    correlation_coefficient, p_value = pearsonr(human_scores, acc_list)
    print('Corr(Human, Exact match)=', correlation_coefficient, p_value)
    correlation_coefficient, p_value = pearsonr(human_scores, pos_div_list)
    print('Corr(Human, Pos. Div.)=', correlation_coefficient, p_value)
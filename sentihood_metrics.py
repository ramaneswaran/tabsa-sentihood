import numpy as np
from sklearn import metrics


"""
To ensure fair evaluation, the methods to compute these metrics are taken from 
"""

def compute_sentihood_aspect_strict_accuracy(test_labels, predicted_labels):
    correct_count = 0
    num_examples = len(test_labels) // 4
    for i in range(num_examples):
        if test_labels[i * 4] == predicted_labels[i * 4]\
                and test_labels[i * 4 + 1] == predicted_labels[i * 4 + 1]\
                and test_labels[i * 4 + 2] == predicted_labels[i * 4 + 2]\
                and test_labels[i * 4 + 3] == predicted_labels[i * 4 + 3]:
            correct_count += 1
    return correct_count / num_examples


def compute_sentihood_aspect_macro_F1(test_labels, predicted_labels):
    total_precision = 0
    total_recall = 0
    num_examples = len(test_labels) // 4
    count_examples_with_sentiments = 0
    for i in range(num_examples):
        test_aspects = set()
        predicted_aspects = set()
        for j in range(4):
            if test_labels[i * 4 + j] != 0:
                test_aspects.add(j)
            if predicted_labels[i * 4 + j] != 0:
                predicted_aspects.add(j)
        if len(test_aspects) == 0:
            continue
        intersection = test_aspects.intersection(predicted_aspects)
        if len(intersection) > 0:
            precision = len(intersection) / len(predicted_aspects)
            recall = len(intersection) / len(test_aspects)
        else:
            precision = 0
            recall = 0
        total_precision += precision
        total_recall += recall
        count_examples_with_sentiments += 1
    ma_P = total_precision / count_examples_with_sentiments
    ma_R = total_recall / count_examples_with_sentiments
    
    if (ma_P + ma_R) == 0:
        return 0
    
    return (2 * ma_P * ma_R) / (ma_P + ma_R)


def compute_sentihood_aspect_macro_AUC(test_labels, scores):
    aspects_test_labels = [[] for _ in range(4)]
    aspects_none_scores = [[] for _ in range(4)]
    for i in range(len(test_labels)):
        if test_labels[i] != 0:
            new_label = 0
        else:
            new_label = 1   # For metrics.roc_auc_score you need to use the score of the maximum label, so "None" : 1
        aspects_test_labels[i % 4].append(new_label)
        aspects_none_scores[i % 4].append(scores[i][0])
    aspect_AUC = []
    for i in range(4):
        aspect_AUC.append(metrics.roc_auc_score(aspects_test_labels[i], aspects_none_scores[i]))
    aspect_macro_AUC = np.mean(aspect_AUC)
    return aspect_macro_AUC


def compute_sentihood_sentiment_classification_metrics(test_labels, scores):
    """Compute macro AUC and accuracy for sentiment classification ignoring "None" scores"""
    # Macro AUC
    sentiment_test_labels = [[] for _ in range(4)]  # One list for each aspect
    sentiment_negative_scores = [[] for _ in range(4)]
    sentiment_predicted_label = []
    sentiment_test_label = []   # One global list
    for i in range(len(test_labels)):
        if test_labels[i] != 0:
            new_test_label = test_labels[i] - 1  # "Positive": 0, "Negative": 1
            sentiment_test_label.append(new_test_label)
            new_negative_score = scores[i][2] / (scores[i][1] + scores[i][2])   # Prob. of "Negative" ignoring "None"
            if new_negative_score > 0.5:
                sentiment_predicted_label.append(1)
            else:
                sentiment_predicted_label.append(0)
            sentiment_test_labels[i % 4].append(new_test_label)
            sentiment_negative_scores[i % 4].append(new_negative_score)
    sentiment_AUC = []
    for i in range(4):
        sentiment_AUC.append(metrics.roc_auc_score(sentiment_test_labels[i], sentiment_negative_scores[i]))
    sentiment_macro_AUC = np.mean(sentiment_AUC)

    # Accuracy
    sentiment_accuracy = metrics.accuracy_score(sentiment_test_label, sentiment_predicted_label)

    return sentiment_macro_AUC, sentiment_accuracy
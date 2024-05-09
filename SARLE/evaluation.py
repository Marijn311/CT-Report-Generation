import numpy as np
import sklearn.metrics
from abnormality_vocabulary import ABNORMALITY_TERMS
import matplotlib.pyplot as plt

def calculate_eval_metrics(predicted_labels, predicted_probs, true_labels):
    """
    Evaluate the performance of the ML based sentence classification in the SARLE-Hybrid variant.
    
    This function calculates the accuracy, AUC, and average precision of the model.
    
    Parameters:
    - predicted_labels (list): The predicted labels for the sentences.
    - predicted_probs (list): The predicted probabilities for the positive class.
    - true_labels (list): The true labels for the sentences.
    
    Returns:
    - accuracy (float): The accuracy of the model.
    - auc (float): The area under the ROC curve.
    - average_precision (float): The average precision score.
    """
    
    # Calculate the performance metrics
    correct_sum = (np.array(true_labels) == np.array(predicted_labels)).sum()
    accuracy = (float(correct_sum)/len(true_labels))
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true = true_labels,
                                     y_score = predicted_probs,
                                     pos_label = 1) 
    auc = sklearn.metrics.auc(fpr, tpr)
    average_precision = sklearn.metrics.average_precision_score(true_labels, predicted_probs)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, auc, average_precision
    
def term_search_performance(radlabel_object):
    """
    Evaluate the performance of the label extraction.

    Args:
        radlabel_object: The RadLabel object containing the predicted and true labels.

    Returns:
        None, but prints the confusion matrix and the performance metrics for each abnormality.

    """
    abnormality_names = list(ABNORMALITY_TERMS.keys())
    for abnormality_name in abnormality_names:
            
        # Get the predicted and true labels for the abnormality presence or absence
        pred_abnormalities = radlabel_object.abnormality_out[abnormality_name].values.tolist() 
        gt_abnormalities = radlabel_object.data.groupby('Filename')[abnormality_name].any() 
        gt_abnormalities = gt_abnormalities.values.astype(int)
        
        assert len(pred_abnormalities)==len(gt_abnormalities), 'Lengths of predicted and true labels are not equal' 
        
        # Calculate the performance metrics
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=gt_abnormalities, y_pred=pred_abnormalities)
        accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1]) / np.sum(confusion_matrix)
        precision = confusion_matrix[1,1] / (confusion_matrix[1,1]+confusion_matrix[0,1])
        recall = confusion_matrix[1,1] / (confusion_matrix[1,1]+confusion_matrix[1,0])
        f1 = 2*(precision*recall) / (precision+recall)
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true = gt_abnormalities, y_score = pred_abnormalities, pos_label = 1)
        auc = sklearn.metrics.auc(fpr, tpr)

        # Print the confusion matrix and the performance metrics
        print(f'\nEvaluation of Abnormality Detection for {abnormality_name}:')
        print('Confusion Matrix:')
        print('\t\tPredicted')
        print('\t\tneg\tpos')
        print('Actual neg\t',confusion_matrix[0,0],'\t',confusion_matrix[0,1])
        print('Actual pos\t',confusion_matrix[1,0],'\t',confusion_matrix[1,1])
    
        print(f'\nAccuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        print(f'AUC: {auc}')

       

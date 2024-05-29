import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, accuracy_score
from config import *

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

def plot_precision_recall_curve(logits, labels, epoch_type):
    """Plot the precision-recall curve (PRC) for multilabel classification.

    Precision-recall curves (PRCs) are typically used in binary classification to study the performance of a classifier.
    The PRC shows the tradeoff between precision and recall for different thresholds. Thresholds are used to binarize the model's output.
    
    Note:
        - Precision = TP / (TP + FP). Precision is about being accurate in your positive predictions.
        - Recall = TP / (TP + FN). Recall is about capturing all the positive instances, not missing any.
        - In multilabel classification, the PRC function can be used by treating each class as a separate binary classification problem.
        - The PRC function is called at the end of the epoch to include all samples in the calculation of the curve.
        - The PRC is particularly useful for imbalanced datasets or when the cost of false positives and false negatives varies.

    Args:
        logits (list of tensors): The model's output logits. There is 1 tensor for each training/validation step in the epoch. Each tensor has shape [batch_size, num_classes].
        labels (list of tensors): The ground truth labels. There is 1 tensor for each training/validation step in the epoch. Each tensor has shape [batch_size, num_classes].
        epoch_type (str): The type of epoch (e.g., "training", "validation").

    Returns:
        float: The macro average AUC (Area Under the Curve) across all classes.

    """
    
    # Unpack the list of tensors to form a single tensor of shape
    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)
    
    # Convert the logits to probabilities. Probalities do NOT sum to 1 because the labels are NOT mutually exclusive.
    probs = torch.sigmoid(logits) 
    
    # Convert torch tensor to numpy array
    labels = labels.detach().clone().cpu().numpy() 
    probs = probs.detach().clone().cpu().numpy()
    
    # To store values per class
    precisions = dict()
    recalls = dict()
    aucs = dict()
    
    # For each class, calculate the precision and recall values, and plot the PRC
    fig, ax = plt.subplots()
    for i in range(NUM_CLASSES):
        assert np.max(labels[:, i]) > 0, f"There are zero positive sample for the {CLASS_NAMES[i]}. The precision-recall cannot be computed due to a division by zero error"
        class_labels = labels[:, i]
        class_probs = probs[:, i]
        precisions[i], recalls[i], _ = precision_recall_curve(y_true=class_labels, probas_pred=class_probs, pos_label=1) 
        ax.plot(recalls[i], precisions[i], lw=2, label=CLASS_NAMES[i]) 
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve for Multilabel Classification')
    ax.legend(loc="lower left") 
    ax.grid(True)
    fig.savefig(f"last_{epoch_type}_precision_recall_curve.png")
    plt.close(fig)    

    # Calculate AUC for each class given the precison and recall values for that class
    for i in range(NUM_CLASSES):
        aucs[i] = auc(recalls[i], precisions[i])
    
    # Compute the macro average AUC across all classes
    macro_avg_auc = sum(aucs.values()) / len(aucs)
 
    if SHOW_PREDICTIONS == True:
        print(f"\n\nNow evaluating PRC for the {epoch_type} epoch.")
        print(f"The AUCs for {CLASS_NAMES} are: {aucs}")
        print(f"The macro averaged AUC is: {macro_avg_auc}")
    
    return macro_avg_auc
    
    
    
def get_accuracy(logits, labels, epoch_type, optimal_thresholds=None):
    """Calculates the classification accuracy.
    
    Notes: 
    -   The accuracy is the number of correct predictions divided by the total number of predictions. 
        This metric very common for classification problems and is very easy to intuitively understand.
    
    -   However the accuracy is not a good metric when the classes are imbalanced. 
        For such cases we can use the precision-recall curve or the AUC.
    
    -   This function calculate the accuracy for the entire epoch using a 0.5 threshold. 
        It also calculates the accuracy using the "optimal threshold". This is the theshold which gave the highest accuracy in the previous traininig epoch.
        For this reason the on_training_epoch_end function is called before the on_validation_epoch_end function (which is not the default in PyTorch Lightning).
    
    Args:
        logits (list of tensors): The model's output logits. There is 1 tensor for each training/validation step in the epoch. Each tensor has shape [batch_size, num_classes].
        labels (list of tensors): The ground truth labels. There is 1 tensor for each training/validation step in the epoch. Each tensor has shape [batch_size, num_classes].
        epoch_type (str): The type of epoch (e.g., "training", "validation").
        optimal_thresholds (dict): The optimal thresholds for each class that were calculated in the previous training epoch.
    
    """
    # Unpack the list of tensors to form a single tensor of shape
    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)
    
    # Convert the logits to probabilities. Probalities do NOT sum to 1 because the labels are NOT mutually exclusive.
    probs = torch.sigmoid(logits)
    
    # Convert torch tensor to numpy array
    labels = labels.detach().clone().cpu().numpy() 
    probs = probs.detach().clone().cpu().numpy()
    
    # To store values per class
    new_optimal_thresholds = dict()
    optimal_accuracies = dict()
    accuracies_05 = dict()
   
    
    # In case of training epoch, calculate the accuracy using 100 different thresholds
    # Save the highest accuracy and the associated threshold. Also save the accuracy using the 0.5 threshold
    if epoch_type == "training":
        for i in range(NUM_CLASSES):
            thresholds = list(np.arange(0.01, 1, 0.01))
            accuracies = [accuracy_score(y_true=labels[:,i], y_pred=(probs[:,i] > t)) for t in thresholds]
            new_optimal_thresholds[i] = thresholds[np.argmax(accuracies)]
            optimal_accuracies[i] = accuracy_score(y_true=labels[:,i], y_pred=(probs[:,i] > new_optimal_thresholds[i]))
            accuracies_05[i] = accuracy_score(y_true=labels[:,i], y_pred=(probs[:,i] > 0.5))
        optimal_thresholds = new_optimal_thresholds # Update the optimal_thresholds variable if a trainig epoch is finished     
        
        
    # Calculate the accuracy using the optimal threshold that was calculated in the previous traininig epoch
    # Also calculate the accuracy using the 0.5 threshold.
    if epoch_type == "validation" or epoch_type == "testing":
        for i in range(NUM_CLASSES):
            optimal_accuracies[i] = accuracy_score(y_true=labels[:,i], y_pred=(probs[:,i] > optimal_thresholds[i]))
            accuracies_05[i] = accuracy_score(y_true=labels[:,i], y_pred=(probs[:,i] > 0.5))
            
    
    # Compute the macro average accuracy across all classes
    macro_avg_optimal_accuracy = sum(optimal_accuracies.values()) / len(optimal_accuracies)
    macro_avg_05_accuracy = sum(accuracies_05.values()) / len(accuracies_05)
    
    
    if SHOW_PREDICTIONS == True:
        print(f"\n\nNow evaluating the ACCURACY for {epoch_type} epoch.")
        print(f"The optimal accuracy thresholds from the last TRAINING epoch for {CLASS_NAMES} are: {optimal_thresholds}")
        print(f"The optimal accuracies for {CLASS_NAMES} are: {optimal_accuracies}")
        print(f"The accuracy using the 0.5 threshold for {CLASS_NAMES} is: {accuracies_05}")

    
    return macro_avg_optimal_accuracy, macro_avg_05_accuracy, optimal_thresholds

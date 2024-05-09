import torch
from config import *


def batch_accuracy(logits, reports):
    """
    Calculate the batch averaged accuracy of the predicted words using teacher forcing during training and validation.

    Args:
        logits (torch.Tensor): The predicted logits of shape (batch_size, target_length, vocab_size).
        reports (torch.Tensor): The ground truth reports of shape (batch_size, target_length).

    Returns:
        accuracy (torch.Tensor): The average accuracy over the batch as a tensor.

    """
    
    # Convert the logits to the most likely token(index) for each word in the report
    pred_idxs = torch.argmax(logits, dim=-1)
    
    # Remove the sos token from the labels 
    labels = reports[:, 1:]

    assert labels.shape[1] == pred_idxs.shape[1], "The labels and predictions should have the same length"

    # To store the accuracy for each sample in the batch
    accuracies = []
    
    # Make a dictionary that maps the tokenized report to the words
    token_to_word = {v: k for k, v in VOCAB_DICT.items()}
    
    # Loop over all the samples in the batch
    for sample in range(BATCH_SIZE):
        
        # Get data associated with the sample
        nr_correct = 0       
        label = labels[sample, :]
        pred_idx = pred_idxs[sample, :]
        report = reports[sample, :]

        # Count the number of correct predictions. Stop counting when the EOS token is reached.
        for i in range(len(label)):
            if label[i] == EOS_IDX:
                break
            if label[i] == pred_idx[i]:
                nr_correct += 1

            # Print the context, ground truth and predicted next word
            if SHOW_PREDICTIONS == True:
                context = report[:i+1]
                gt_next_word = report[i+1]
                pred_next_word = pred_idx[i]
                gt_next_word = gt_next_word.unsqueeze(0)
                pred_next_word = pred_next_word.unsqueeze(0)
                untokenised_context = [token_to_word[word.item()] for word in context]
                untokenised_gt_next_word = token_to_word[gt_next_word.item()]
                untokenised_pred_next_word = token_to_word[pred_next_word.item()]
                print(f"\nAvailable context: {untokenised_context}")
                print(f"GT next word: {untokenised_gt_next_word}")
                print(f"Predicted next word: {untokenised_pred_next_word}")

        # Calculate the accuracy for the sample
        accuracy = (nr_correct / i) 
        accuracies.append(accuracy)

    # Calculate the average accuracy over the batch
    accuracy = sum(accuracies) / len(accuracies)
    accuracy = torch.tensor(accuracy, device='cuda')

    return accuracy
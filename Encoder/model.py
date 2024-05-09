import pytorch_lightning as pl
from torch import nn
import torch
import numpy as np
from config import *
import einops
from model_architectures.ct_net import ct_net_architecture
from model_architectures.medical_net import medical_net_architecture
from model_architectures.convs_3d_classifier import convs_3d_classifier
from model_architectures.attention_pooling_classifier import attention_pooling_classifier
from model_architectures.transformer_attention_classifier import transformer_attention_classifier
import wandb
from metrics import *

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class model(pl.LightningModule):
    """A model class for a PyTorch Lightning model.
    
    This class defines the structure and behavior of the model used during fitting, evaluation, testing. 
    It includes methods for configuring optimizers, defining the forward pass, calculating the loss function, and handling training and validation steps and epoch ends.

    Atributes:
        model (nn.Module): The neural network model/arhitecture used for training and evaluation.
        loss_function (nn.Module): The loss function used to calculate the model's loss.
        train_step_outputs (list): A list of dictionaries containing the model's outputs for each training step.
        validation_step_outputs (list): A list of dictionaries containing the model's outputs for each validation step.
        test_step_outputs (list): A list of dictionaries containing the model's outputs for each test step.
        optimal_thresholds (dict): A dictionary containing the optimal thresholds for each class, as calculated in the previous training epoch.
        
    Methods:
        __init__: Initializes the model architecture. 
        configure_optimizers: Configures the optimizer used for training the model.
        forward: Defines the forward pass of the model.
        training_step: Defines the behavior of the model during a training step.
        validation_step: Defines the behavior of the model during a validation step.
        test_step: Defines the behavior of the model during a test step.
        on_train_epoch_end: Defines the behavior of the model at the end of a training epoch.
        on_validation_epoch_end: Defines the behavior of the model at the end of a validation epoch. This function first calls the on_train_epoch_end function to change the order of execution of the two functions. (By default PyTorch Lightning calls the on_validation_epoch_end function before the on_training_epoch_end function.)
        on_test_epoch_end: Defines the behavior of the model at the end of a test epoch.
    """

    def __init__(self): 
        super().__init__()
        
        # Define the model architecture
        if ARCHITECTURE == "ct_net" and FILE_TYPE == "image":
            self.model = ct_net_architecture()
            print("CT-Net architecture loaded")
           
        if ARCHITECTURE == "medical_net" and FILE_TYPE == "image":
            self.model = medical_net_architecture()
            print("Medical-Net architecture loaded")
        
        if ARCHITECTURE == "ct_net" and FILE_TYPE == "feature_tensor" and CLASSIFIER == "convs_3d":
            self.model = convs_3d_classifier()
            print("3D-Convolutional classifier loaded")
            
        if ARCHITECTURE == "ct_net" and FILE_TYPE == "feature_tensor" and CLASSIFIER == "attention_pooling":
            self.model = attention_pooling_classifier()
            print("Attention pooling classifier loaded")
    
        if ARCHITECTURE == "ct_net" and FILE_TYPE == "feature_tensor" and CLASSIFIER == "transformer_attention":
            self.model = transformer_attention_classifier()
            print("Transformer attention classifier loaded")
            
        if ARCHITECTURE == "medical_net" and FILE_TYPE == "feature_tensor" and CLASSIFIER == "convs_3d":
            self.model = convs_3d_classifier()
            print("3D-Convolutional classifier loaded")

        
        # Define the loss function
        self.loss_function = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=POSITIVE_WEIGHTS) 
        
        # To store the outputs of the training/validation/test steps. Used to calculate epoch averaged metrics
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Initialize the optimal thresholds for each class with a dummy value. These will be updated at the end of each training epoch.
        self.optimal_thresholds = {i: 0.5 for i in range(NUM_CLASSES)}
        
        # Initialize a Weights & Biases run
        wandb.init( 
        mode='offline', 
        project="report_experiments_encoder",  
        config={
            "image_file_name": IMAGE_FILE_NAME,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "grad_accum": GRAD_ACCUM,
            "positive_weights": POSITIVE_WEIGHTS,
            "class_names": CLASS_NAMES,
            "architecture": ARCHITECTURE,
            "classifier": CLASSIFIER,
            "data_augment": DATA_AUGMENT,
            "batch_size": BATCH_SIZE,
            "random_seed": RANDOM_SEED,
            "stage": STAGE,

        }) 
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return {"optimizer": optimizer, "monitor": "val_loss"}


    def forward(self, x):
        logits = self.model(x)
        return logits
    
    
    def training_step(self, batch, batch_idx):
        """Perform a training step for the model."""
    
        images, labels = batch           
           
        logits = self.forward(images)

        loss = self.loss_function(logits, labels)
        
        self.train_step_outputs.append( {'loss': loss, 'labels': labels, 'logits': logits} ) 
        
        wandb.log({'training_loss_step': loss})
        
        # Print the probabilities and the ground truth labels. Useful for monitoring model behavior.
        if SHOW_PREDICTIONS == True:
            probs = torch.sigmoid(logits) 
            probs = probs.cpu().detach().numpy().flatten()
            labels = labels.cpu().detach().numpy().flatten()
            probs = einops.rearrange(probs, '(b c) -> b c', b=BATCH_SIZE, c=NUM_CLASSES)
            labels = einops.rearrange(labels, '(b c) -> b c', b=BATCH_SIZE, c=NUM_CLASSES)
            formatted_probs = np.vectorize(lambda x: f"{x:.2f}")(probs)
            print(f"The probilities are: {formatted_probs.squeeze()}, the actual labels are: {labels.squeeze()}")  
        
        return {'loss': loss}

    
    def validation_step(self, batch, batch_idx):
        """Perform a validation step for the model."""
        
        images, labels = batch           
           
        logits = self.forward(images)

        loss = self.loss_function(logits, labels)
      
        self.validation_step_outputs.append( {'loss': loss, 'labels': labels, 'logits': logits} )
        
        wandb.log({'validation_loss_step': loss})
               
        # Print the probabilities and the ground truth labels. Useful for monitoring model behavior.
        if SHOW_PREDICTIONS == True:
            probs = torch.sigmoid(logits) 
            probs = probs.cpu().detach().numpy().flatten()
            labels = labels.cpu().detach().numpy().flatten()
            probs = einops.rearrange(probs, '(b c) -> b c', b=BATCH_SIZE, c=NUM_CLASSES)
            labels = einops.rearrange(labels, '(b c) -> b c', b=BATCH_SIZE, c=NUM_CLASSES)
            formatted_probs = np.vectorize(lambda x: f"{x:.2f}")(probs)
            print(f"The probilities are: {formatted_probs.squeeze()}, the actual labels are: {labels.squeeze()}") 
        
        return {'loss': loss}



    def test_step(self, batch, batch_idx):
        """Perform a test step for the model."""

        images, labels = batch           
                
        logits = self.forward(images)

        loss = self.loss_function(logits, labels)      
        
        self.test_step_outputs.append( {'loss': loss, 'labels': labels, 'logits': logits} )
        
        # Print the probabilities and the ground truth labels. Useful for monitoring model behavior.
        if SHOW_PREDICTIONS == True:
            probs = torch.sigmoid(logits) 
            probs = probs.cpu().detach().numpy().flatten()
            labels = labels.cpu().detach().numpy().flatten()
            probs = einops.rearrange(probs, '(b c) -> b c', b=BATCH_SIZE, c=NUM_CLASSES)
            labels = einops.rearrange(labels, '(b c) -> b c', b=BATCH_SIZE, c=NUM_CLASSES)
            formatted_probs = np.vectorize(lambda x: f"{x:.2f}")(probs) 
            print(f"The probilities are: {formatted_probs.squeeze()}, the actual labels are: {labels.squeeze()}")
        
        return {'loss': loss}
    
    
    def on_train_epoch_end(self, validation_initiated=False):
        """Perform actions at the end of a training epoch.
        When on_validation_epoch_end is initiated by the lightning trainer, it first calls this on_train_epoch_end function with validation_initiated=True. 
        In that case the code below will be executed.
        
        Next the lightning trainer calls the on_training_epoch_end this time with validation_initiated=False, 
        in which case nothing happens because the training_epoch_end code shouldn't be executed twice in a row. 
        """
        if validation_initiated == True: 
            # Load all values that were saved during the trainig steps
            loss_values = [item['loss'] for item in self.train_step_outputs]
            labels_values = [item['labels'] for item in self.train_step_outputs]
            logits_values = [item['logits'] for item in self.train_step_outputs]
            
            # Calculate the epoch average metrics
            epoch_loss = torch.stack(loss_values).mean() 
            epoch_macro_avg_prc_auc  = plot_precision_recall_curve(logits_values, labels_values, "training")
            epoch_macro_avg_optimal_accuracy, epoch_macro_avg_05_accuracy, optimal_thresholds = get_accuracy(logits_values, labels_values, "training", self.optimal_thresholds)
            
            # Update the optimal thresholds so the can be used in the validation epoch
            self.optimal_thresholds = optimal_thresholds
            
            # Clear the data for the next epoch
            self.train_step_outputs.clear()  
            
            # Needed for the callback that saves model checkpoints
            self.log(f'macro_avg_optimal_accuracy', epoch_macro_avg_optimal_accuracy)
            
            
            # Log metrics to Weights & Biases
            wandb.log({
                'train_loss_epoch_average': epoch_loss,
                'train_macro_avg_optimal_accuracy_epoch_average': epoch_macro_avg_optimal_accuracy,
                'train_macro_avg_05_accuracy_epoch_average': epoch_macro_avg_05_accuracy, 
                'train_macro_avg_prc_auc_epoch_average': epoch_macro_avg_prc_auc,
                })
                    
            # Print the epoch averaged metrics
            print(f'\n\nTrain loss (average of last epoch): {epoch_loss:.4f}') 
            print(f'\n\nTrain 0.5 accuracy (average of last epoch): {epoch_macro_avg_05_accuracy:.3f}')
            print(f'\n\nTrain optimal accuracy (average of last epoch): {epoch_macro_avg_optimal_accuracy:.3f}')
            print(f'\n\nTrain macro averaged prc_auc (average of last epoch): {epoch_macro_avg_prc_auc:.3f}')
        
    
    def on_validation_epoch_end(self):
        """Perform actions at the end of a validation epoch."""
        
        if STAGE == "fit":
            self.on_train_epoch_end(validation_initiated=True)
        
        # Load all values that were saved during the trainig steps
        loss_values = [item['loss'] for item in self.validation_step_outputs]
        labels_values = [item['labels'] for item in self.validation_step_outputs]
        logits_values = [item['logits'] for item in self.validation_step_outputs]
        
        # Calculate the epoch average metrics
        epoch_loss = torch.stack(loss_values).mean()
        epoch_macro_avg_prc_auc  = plot_precision_recall_curve(logits_values, labels_values, "validation")
        epoch_macro_avg_optimal_accuracy, epoch_macro_avg_05_accuracy, _ = get_accuracy(logits_values, labels_values, "validation", self.optimal_thresholds)
        
        # Clear the data for the next epoch
        self.validation_step_outputs.clear()  

        # Needed for the callback that saves model checkpoints
        self.log('val_loss', epoch_loss) 
        
        # Log metrics to Weights & Biases
        wandb.log({
            'val_loss_epoch_average': epoch_loss,
            'val_macro_avg_optimal_accuracy_epoch_average': epoch_macro_avg_optimal_accuracy,
            'val_macro_avg_05_accuracy_epoch_average': epoch_macro_avg_05_accuracy,
            'val_macro_avg_prc_auc_epoch_average': epoch_macro_avg_prc_auc,
            })
        
        # Print the epoch averaged metrics
        print(f'\n\nValidation loss (average of last epoch): {epoch_loss:.4f}') 
        print(f'\n\nValidation 0.5 accuracy (average of last epoch): {epoch_macro_avg_05_accuracy:.3f}')
        print(f'\n\nValidation optimal accuracy (average of last epoch): {epoch_macro_avg_optimal_accuracy:.3f}')
        print(f'\n\nValidation macro averaged prc_auc (average of last epoch): {epoch_macro_avg_prc_auc:.3f}')

    

    def on_test_epoch_end(self):
        """Perform actions at the end of a test epoch."""	
        
        # Load all values that were saved during the test steps
        loss_values = [item['loss'] for item in self.test_step_outputs]
        labels_values = [item['labels'] for item in self.test_step_outputs]
        logits_values = [item['logits'] for item in self.test_step_outputs]
        
        # Calculate the epoch average metrics
        epoch_loss = torch.stack(loss_values).mean()
        epoch_macro_avg_prc_auc  = plot_precision_recall_curve(logits_values, labels_values, "testing")
        epoch_macro_avg_optimal_accuracy, epoch_macro_avg_05_accuracy, _ = get_accuracy(logits_values, labels_values, "testing", self.optimal_thresholds)
    
        # Clear the data for the next epoch
        self.test_step_outputs.clear()  
        
        # Log metrics to Weights & Biases
        wandb.log({
            'test_loss_epoch_average': epoch_loss,
            'test_macro_avg_optimal_accuracy_epoch_average': epoch_macro_avg_optimal_accuracy,
            'test_macro_avg_05_accuracy_epoch_average': epoch_macro_avg_05_accuracy, 
            'test_macro_avg_prc_auc_epoch_average': epoch_macro_avg_prc_auc,
            })
        
        # Print the epoch averaged metrics
        print(f'\n\nTest loss (average of last epoch): {epoch_loss:.4f}') 
        print(f'\n\nTest 0.5 accuracy (average of last epoch): {epoch_macro_avg_05_accuracy:.3f}')
        print(f'\n\nTest optimal accuracy (average of last epoch): {epoch_macro_avg_optimal_accuracy:.3f}')
        print(f'\n\nTest macro averaged prc_auc (average of last epoch): {epoch_macro_avg_prc_auc:.3f}')




     
            



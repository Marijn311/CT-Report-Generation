import torch
import pytorch_lightning as pl
from torch import nn
from config import *
import wandb
from model_architectures.transformer_scratch import transformer_scratch_architecture
from model_architectures.bioclinical_bert import bioclinical_bert_decoder_architecture
from model_architectures.biogpt import biogpt_architecture
from utils import *
import numpy as np
import einops

#Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class model(pl.LightningModule):
    """ A model class for training and evaluating a PyTorch Lightning model.
    
    This class defines the structure and behavior of the model used for training and evaluation. 
    It includes methods for configuring optimizers, defining the forward pass, calculating the loss function, and handling training and validation steps.

    Attributes:
        train_step_outputs (list): A list to store the output of each training step.
        validation_step_outputs (list): A list to store the output of each validation step.

    Methods:
        __init__(): Sets up the architecture and initializes a Weights & Biases run to save the training logs.
        loss_function(logits, reports): Calculates the loss function given the model's predictions and the target reports.
        configure_optimizers(): Configures the optimizer used for training.
        forward(images, reports): Performs the forward pass of the model.
        training_step(batch, batch_idx): Defines the behavior of a training step.
        validation_step(batch, batch_idx): Defines the behavior of a validation step.
        on_train_epoch_end(): Performs actions at the end of each training epoch, such as calculation epoch average metrics and logging to Weights & Biases.
        on_validation_epoch_end(): Performs actions at the end of each validation epoch, such as calculation epoch average metrics and logging to Weights & Biases.
    
    """
    
    
    
    
    def __init__(self):
        
        super().__init__()

        # Initialize a Weights & Biases run to save the training logs
        wandb.init(
        mode='offline', 
        project="report_experiments_decoder",  
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,	
            "architecture": ARCHITECTURE,
            "dataset": HOSPITAL_NAMES,
            "class_names": CLASS_NAMES,
            "image_dir": IMAGE_DIR,
            "max_length": MAX_LENGTH,
            "stage": STAGE,
        })

        # Initialize the model architecture based on the specified configuration
        if ARCHITECTURE == 'transformer_scratch':   
            self.model = transformer_scratch_architecture()
            print("Using transformer from scratch architecture")
        elif ARCHITECTURE == 'bioclinical_bert':
            self.model = bioclinical_bert_decoder_architecture()
            print("Using bioclinical BERT decoder architecture")
        elif ARCHITECTURE == 'bio_gpt':
            self.model = biogpt_architecture()
            print("Using bio_gpt decoder architecture")        
        
        #to store the output of the model steps during the epoch
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
            
    
    def loss_function(self, logits, reports):
        """Calculates the loss function (Cross Entropy Loss) given the model's predictions and the target reports."""
        # Remove the SOS token from the labels
        labels = reports[:, 1:] 
        assert labels.shape[1] == logits.shape[1], "The labels and predictions should have the same length"

        # Reshape (flatten) the logits and labels to match the CrossEntropyLoss function
        labels = einops.rearrange(labels, 'batch seq_length -> (batch seq_length)').long()
        logits = einops.rearrange(logits, 'batch seq_length vocab_size -> (batch seq_length) vocab_size').float()
        cel = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Ignore the padding when calculating the loss
        loss = cel(logits, labels)
        
        return loss



    def configure_optimizers(self):
        """ Configures the optimizer used for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE) 
        return {"optimizer": optimizer, "monitor": "val_loss" }
    
    
    
    def forward(self, images, reports): 
        """Does the forward pass of the model."""
        logits = self.model(images, reports)
        return logits



    def training_step(self, batch, batch_idx):
        """Defines the behavior of a training step."""
        
        images, reports = batch                     
        
        logits = self.forward(images, reports)
        
        assert logits.shape[1] == MAX_LENGTH-1, "MAX_LENGTH is originally defined based on the labels (reports) and it includes the SOS token. The model should NOT predict and SOS token so it should return MAXLENGTH-1 predicted tokens"      
        
        loss = self.loss_function(logits, reports)

        accuracy = batch_accuracy(logits, reports)
        
        print("\nTraining accuracy (batch average): {:.2f}".format(accuracy.item()))

        self.train_step_outputs.append( {'loss': loss, 'accuracy': accuracy} )
        
        wandb.log({ 'train_loss_step': loss,
                    'train_accuracy_step': accuracy,})
                         
        return {'loss': loss}

        
    def validation_step(self, batch, batch_idx):
        """Defines the behavior of a validation step."""
        
        images, reports = batch                     
        
        logits = self.forward(images, reports)  
        
        assert logits.shape[1] == MAX_LENGTH-1, "MAX_LENGTH is originally defined based on the labels (reports) and it includes the SOS token. The model should NOT predict and SOS token so it should return MAXLENGTH-1 predicted tokens"      
       
        loss = self.loss_function(logits, reports)

        accuracy = batch_accuracy(logits, reports)
        
        print("\nValidation accuracy (batch average): {:.2f}".format(accuracy.item()))

        self.validation_step_outputs.append( {'loss': loss, 'accuracy': accuracy} )
        
        wandb.log({ 'val_loss_step': loss,
                    'val_accuracy_step': accuracy, })
        
        return {'loss': loss}
        
        
        
    def on_train_epoch_end(self):
        """Defines the behavior at the end of each training epoch."""
        
        # Calculate the epoch average loss and accuracy
        loss_values = [item['loss'] for item in self.train_step_outputs]
        accuracy_values = [item['accuracy'] for item in self.train_step_outputs]
        epoch_loss = torch.stack(loss_values).mean()
        epoch_accuracy = torch.stack(accuracy_values).mean()
      
        # Clear the data for the next epoch
        self.train_step_outputs.clear()  

        self.log('train_accuracy', epoch_accuracy) #todo i dont think this is used anymore
        
        wandb.log({
            'train_loss_epoch_average': epoch_loss,
            'train_accuracy_epoch_average': epoch_accuracy,
            })
                
        print(f'\nTrain loss (average of last epoch): {epoch_loss:.2f}')
        print(f'\nTrain accuracy with teacher forcing (average of last epoch): {epoch_accuracy:.2f}')
   
        
    def on_validation_epoch_end(self):
        """"Defines the behavior at the end of each validation epoch."""
        
        # Calculate the epoch average loss, accuracy 
        loss_values = [item['loss'] for item in self.validation_step_outputs]
        accuracy_values = [item['accuracy'] for item in self.validation_step_outputs]
        epoch_loss = torch.stack(loss_values).mean() #loss values are tensors so we stack them and take the mean
        epoch_accuracy = torch.stack(accuracy_values).mean()
        
        # Clear the data for the next epoch
        self.validation_step_outputs.clear()  

        # Needed for the callback that saves the model checkpoints
        self.log('val_loss', epoch_loss) 
        self.log('val_accuracy', epoch_accuracy) 

        wandb.log({
            'val_loss_epoch_average': epoch_loss,
            'val_accuracy_epoch_average': epoch_accuracy,
            })
        
        print(f'\nValidation loss (average of last epoch): {epoch_loss:.2f}')
        print(f'\nValidation accuracy (average of last epoch): {epoch_accuracy:.2f}')
   











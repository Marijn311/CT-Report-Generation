import torch.nn as nn
import einops 
from config import *
import pytorch_lightning as pl

# Set seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class attention_pooling_classifier(pl.LightningModule):
    """Class for the attention pooling classifier.
     This classifier takes the feature map tensors and uses linear layers and an attention pooling mechanism
     to classify the inputted feature tensors into output classes.
    
    The CT-Net feature extractor (which came before this classifier) outputs a feature tensor for each 3D image.
    This feature tensor consists of multiple smaller feature tensors, one for each chunk of 3 transverse slices of the original 3D image. 
    Each chunk is called a "token". The entire 3D image is therefore called a "sequence of tokens".
    The chunks dimension is thus called "sequence_length". 
    
    The CT-Net feature tensors have shape (batch, sequence_length, channels=512, f1, f2) when the are inputted. Where f1 and f2 are the spatial dimensions of the feature maps.
    
    This classifer flattens the f (feature) dimensions into a feature vector. 
    Next it reduces reduces the channel dimension by averaging over the channel dimension.
    The preproccesed feature tensor has shape (batch, sequence_length, feature_vector_size).
    
    This classifier consists of a few main components:
    1. A linear layer with a non-linear activation function and dropout. This layer does not reshape the input, it is just some non-linear transformation with trainable parameters.
    2. An attention pooling layer. 
            This layer calculates an attention score (single value) for every feature vector in sequence_length. 
            NOTE that the attention mechanism multiplies a feature vector with itself and not with other feature vectors. 
            This NOT a self-attention mechanism like in transformers. 
            Rather this is a mechanism to learn which parts of a feature vector are more important for making predictions. 
            The attention scores are passed through a softmax layer to normalize the scores to sum to 1. 
            An attention-weighted averaging (pooling) is aplied to output 1 feature vector that represents the entire sequence.
    3. The weighted/pooled feature vector is passed through a final linear layer to get the final prediction.
    
    Attributes:
        first_linear_layer (nn.Sequential): Sequential container for the first linear layer.
        attention_a (nn.Sequential): Sequential container for the first attention layer.
        attention_b (nn.Sequential): Sequential container for the second attention layer.
        attention_c (nn.Linear): Linear layer for the final attention layer.
        attention_softmax (nn.Softmax): Softmax layer to normalize the attention scores.
        final_linear_layer (nn.Linear): Linear layer for the final prediction.

    Methods:
        forward(x): Performs the forward pass of the model.
        
    
    """
    
    def __init__(self):
        super().__init__()
        
        # Define the feature vector size
        feature_vector_size = IMAGE_SIZE[-1]*IMAGE_SIZE[-2]

        # Define the first linear layer
        self.first_linear_layer = nn.Sequential(
            nn.Linear(feature_vector_size, feature_vector_size),
            nn.ReLU(),
            nn.Dropout(0.25),
        ).to("cuda")
  
        # Define the attention pooling layers
        self.attention_a = nn.Sequential(nn.Linear(feature_vector_size, 1024), nn.Tanh(), nn.Dropout(0.25)).to("cuda")
        self.attention_b = nn.Sequential(nn.Linear(feature_vector_size, 1024), nn.Sigmoid(), nn.Dropout(0.25)).to("cuda")
        self.attention_c = nn.Linear(*[1024, 1]).to("cuda") 
        self.attention_softmax = nn.Softmax(dim=0).to("cuda")
        
        # Define the final linear layer
        self.final_linear_layer = nn.Linear(feature_vector_size, NUM_CLASSES).to("cuda")
        

    def forward(self, x):
        """Performs the forward pass of the model."""
        
        # Preprocess the input tensor by reducing the channel dimension and flattening the feature dimensions
        x = einops.reduce(x, 'b chunks c f1 f2 -> b chunks f1 f2', 'mean')
        x = einops.rearrange(x, 'b chunks f1 f2 -> b chunks (f1 f2)')
    
        # Pass the feature tensor through the first linear layer. (This doesnt change the shape of the tensor, just applies a trainable non-linear transformation)
        x = self.first_linear_layer(x)

        # Get the normalized attention scores for each token in the sequence
        a = self.attention_a(x) 
        b = self.attention_b(x)
        a_x_b = a.mul(b)
        attention_logits = self.attention_c(a_x_b)
        attention = self.attention_softmax(attention_logits) 
      
        # Use the attention scores as weights to perform a weighted averaging of the tokens (feature vectors) in the sequence. The output is a single feature vector.
        weighted_data = x * attention  
        weighted_avg = einops.reduce(weighted_data, 'b t x -> b x', 'mean' ) 
        
        # Pass the weighted average feature tensor through a the final linear layer to get the final prediction.
        logits = self.final_linear_layer(weighted_avg) 
        
        return logits

    


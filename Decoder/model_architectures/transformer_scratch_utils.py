import torch
import pytorch_lightning as pl
from torch import nn
from config import *
import einops

"""This file contains the building blocks (in the form of multiple classes) for the transformer model from scratch."""

class SelfAttention(pl.LightningModule):
    """Self-Attention module that performs multi-head attention mechanism.

    Attributes:
        head_dim (int): The dimension of the embedding vector that is processed per head.
        values (nn.Linear): Linear layer for values in the multi-head attention.
        keys (nn.Linear): Linear layer for keys in the multi-head attention.
        queries (nn.Linear): Linear layer for queries in the multi-head attention.
        fc_out (nn.Linear): Fully connected layer to concatenate the QVK at the end of the attention block.

    Methods:
        forward(values, keys, query, mask): Performs the forward pass of the self-attention module.

    """
    def __init__(self):
        super().__init__()

        # Determine the dimension of the embedding vector that is processed per head
        self.head_dim = HIDDEN_SIZE // NR_HEADS 

        # Create the three linear layers for the VKQ, See the "Multi-Head Attention" diagram on the 4th page of the "attention is all you need" paper
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False).to('cuda')
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).to('cuda')
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to('cuda')

        # Fully connected layer to concatenate the QVK at the end of the "scaled Dot-Product Attention" block
        self.fc_out = nn.Linear(NR_HEADS*self.head_dim, HIDDEN_SIZE).to('cuda')


    def forward(self, values, keys, query, mask):
        """Performs the forward pass of the self-attention module."""
        
        # Split the embeding vectors into NR_HEADS amount of equal chuncks so they can be processed in parallel by the multiple attention heads 
        values = einops.rearrange(values, 'b v (h d) -> b v h d', h=NR_HEADS) #b for batch, v for value_len, h for NR_HEADS, d for head_dim
        keys = einops.rearrange(keys, 'b k (h d) -> b k h d', h=NR_HEADS) 
        queries = einops.rearrange(query, 'b q (h d) -> b q h d', h=NR_HEADS)
       
        # Pass the embeddings of the keys, values, and queries through the linear layers to create learnable embedding that better seperate the words that often occur in the same context
        values = self.values(values)
        keys = self.keys(keys) 
        queries = self.queries(queries) 

        # Perform the "scaled dot-product attention" block of the AIAYN paper
            #-Multiply the queries with the keys (first step in calculating the attention). Tthe attention formula is softmax( QK^T \ sqr(D_k) ) * V
            #-Ddefine "energy" as QK^T (the dot product between the queries and the transpose of the keys)
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) #b for batch, q for query_len, h for NR_HEADS, d for head_dim, k for key_len 
        
        # In case of masked attention, set the energy to -infinity where the mask is zero
        if mask is not None:
            mask=mask.to('cuda')
            if PRECISION == 16:
                neg_infinity = float("-65504") #the largest negative number for 16 precision 
            if PRECISION == 32:
                neg_infinity = float("-1e20") #an even larger negative number for 32 precision 
            energy = energy.masked_fill(mask == 0, neg_infinity) 
            
        # Calculate the (masked) attention by dividing the energy by the squareroot of the HIDDEN_SIZE and taking the softmax of that, and finally multiplying with the values
        attention_precursor = torch.softmax(energy / HIDDEN_SIZE**(1/2), dim=3) 
        attention = torch.einsum("bhql,blhd->bqhd", [attention_precursor, values])
        
        # Concatenate the attention scores of the NR_HEADS back into a single value at the end of the attention block
        attention = einops.rearrange(attention, 'b q h d -> b q (h d)') #b for batch, q for query_len, h for NR_HEADS, d for head_dim
       
        # Final linear layer that comes after the attention block and the concat. See figure 2 in the "attention is all you need" paper
        out = self.fc_out(attention)
       
        return out
    

class TransformerBlock(pl.LightningModule):
    """This class represents a Transformer block on the encoder side.
    It implements the "nx" block described in Figure 1 of the "Attention is All You Need" paper.

    Attributes:
        attention: The self-attention block as defined in another class.
        norm1: The first normalization layer.
        norm2: The second normalization layer.
        feed_forward: The feed-forward block, consisting of 2 linear layers which scale with FORWARD_EXPANSION.
        dropout: The dropout module used for regularization.

    Methods:
        forward: Performs the forward pass of the Transformer block.

    """

    def __init__(self):
        super().__init__()
        
        # Define the building blocks of the transformer block
        self.attention = SelfAttention().to('cuda')
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE).to('cuda') 
        self.norm2 = nn.LayerNorm(HIDDEN_SIZE).to('cuda') 
        self.feed_forward = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, FORWARD_EXPANSION*HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FORWARD_EXPANSION*HIDDEN_SIZE, HIDDEN_SIZE)     
        ).to('cuda')
        self.dropout = nn.Dropout(DROPOUT).to('cuda')
    
    def forward(self, value, key, query, mask):
        """Performs the forward pass of the Transformer block."""
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x) 
        out = self.dropout(self.norm2(forward + x)) 
        
        return out
    
class Encoder(pl.LightningModule):
    """The Encoder class is responsible for reshaping the encoded images into the desired shape of (batch_size, seq_len, hidden_size).
    It supports two different representations: 'feature_map' and 'token'.

    Attributes:
        reshape_to_hidden_size (nn.Linear): Linear layer used to reshape the feature maps into the desired shape.
        img_embedding (nn.Embedding): Embedding layer used to embed the integer values of the image tokens.

    Methods:
        forward(images): Performs the forward pass of the encoder.

    """
    
    def __init__(self):
        super().__init__()

        # To reshape the images into the desired shape
        if REPRESENTATION == 'feature_map':
            self.reshape_to_hidden_size = nn.Linear(IMAGE_SIZE[-1]*IMAGE_SIZE[-2], HIDDEN_SIZE).to('cuda') 
        if REPRESENTATION == 'token':
            self.img_embedding = nn.Embedding(NR_IMG_TOKENS, HIDDEN_SIZE).to('cuda') 


    def forward(self, images): 
        """Performs the forward pass of the encoder."""
        
        if REPRESENTATION == 'feature_map':
            images = einops.rearrange(images, 'b c f1 f2 -> b c (f1 f2)')                               # Flatten the features maps
            images = self.reshape_to_hidden_size(images)                                                # Reshape the flatted featuremap to HIDDEN_SIZE
            #todo is a normalisation or activation function needed after this linear layer to make sure the image embeddings are in a certain range?
            
        if REPRESENTATION == 'token':
            min_value = 0
            max_value = 40                                                                              # This may change depending on the encoder model and dataset that was used. Set SHOW_DATA_EXAMPLES to True in config.py to see the min and max values of the encoded images. Choose a range that covers most of the values.  
            images = torch.clamp(images, min=min_value, max=max_value)                                  # Clamp the encoded images (which are logits values from a ReLu layer) to a fixed range
            scaled_values = (((images - min_value) / (max_value - min_value)) * (NR_IMG_TOKENS-1))      # Scale the clamped values to the range [0, NR_IMG_TOKENS-1]
            integer_values = torch.round(scaled_values).to(torch.int)                                   # Round the scaled values to integers. The "integer values" are effectively the image tokens.
            images = self.img_embedding(integer_values)                                                 # Embed the integer values to the hidden size of the model

        return images



class DecoderBlock(pl.LightningModule):
    """This class represents a single transformer "nx" block on the decoder side of the transformer model.
    
    The decoder block consists of two main components: self-attention and cross-attention.
    It takes an input tensor 'x' (the embedded reports) and performs self-attention on it using the `SelfAttention` module.
    In the cross-attention block, an attention step is performed between the processed 'x' with the input tensors `value` and `key` which represent the encoded image.
    
    Attributes:
        attention (SelfAttention): The self-attention block as defined in another class.
        norm (nn.LayerNorm): A normalization layer.
        transformer_block (TransformerBlock): The transformer block as defined in another class.
        dropout (nn.Dropout): The dropout module used for regularization.
        
    Returns:
        out (torch.Tensor): The output tensor of the transformer block.
    """
    def __init__(self):
        super().__init__()

        # Define the building blocks of the decoder block
        self.attention = SelfAttention() 
        self.norm = nn.LayerNorm(HIDDEN_SIZE).to('cuda')
        self.transformer_block = TransformerBlock()
        self.dropout = nn.Dropout(DROPOUT).to('cuda')

    def forward(self, x, value, key, report_mask):
        """Forward pass of the decoder block.
        
        Args:
            x (torch.Tensor): The input tensor (ground truth reports) to the decoder block.
            value (torch.Tensor): The value tensor (encoded images) for cross-attention.
            key (torch.Tensor): The key tensor (encoded images) for cross-attention.
            report_mask (torch.Tensor): The mask tensor to mask future words in the report.
        
        Returns:
            out (torch.Tensor): The output tensor after applying self-attention and cross-attention.
        """
        attention = self.attention(x, x, x, report_mask)
        query = self.norm(attention + x)
        query = self.dropout(query)
        
        out = self.transformer_block(value, key, query, mask=None)
        
        return out
    
class Decoder(pl.LightningModule):
    """This is the entire decoder which uses the decoder block and does the initial embedding of the target reports
    and the final linear layer to get the logits for the predicted words.

    Attributes:
        word_embedding (nn.Embedding): Word embedding layer for target reports.
        positional_embedding (nn.Embedding): Positional embedding layer for target reports.
        decoder_blocks (nn.ModuleList): Stack of decoder blocks, as defined in another class.
        fc_out (nn.Linear): Final linear layer for generating logits.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self):
        super().__init__()
        
        # Define functions for word embeddings and for positional embedding
        self.word_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE).to('cuda')
        self.positional_embedding = nn.Embedding(MAX_LENGTH, HIDDEN_SIZE).to('cuda')

        # Stack the decoder blocks 
        self.decoder_blocks = nn.ModuleList([
                DecoderBlock()
            for _ in range(NUM_LAYERS)]) 
        
        # Define the final linear layer and a dropout function
        self.fc_out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE).to('cuda')
        self.dropout = nn.Dropout(DROPOUT).to('cuda')

    def forward(self, reports, images, report_mask):
        """Performs the forward pass of the decoder."""
        
        # Create the word and positional embeddings for the target reports
        target_length = MAX_LENGTH-1 
        positions = torch.arange(0, target_length).expand(BATCH_SIZE, target_length).to('cuda')
        word_embeddings = self.word_embedding(reports)
        positional_embeddings = self.positional_embedding(positions)          
        embedded_reports = self.dropout(word_embeddings + positional_embeddings)

        # Pass the embedded reports qnd encoded images through the decoder blocks
        for decoder_block in self.decoder_blocks:
            embedded_reports = decoder_block(embedded_reports, images, images, report_mask)
        
        # Pass the output of the decoder blocks through the final linear layer to get shape (batch, target_length, vocab_size)
        logits = self.fc_out(embedded_reports) 

        return logits
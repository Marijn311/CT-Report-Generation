import pytorch_lightning as pl
from torch import nn
from config import *
from transformers import AutoModel
import einops
from transformers.models.bert.modeling_bert import BertOnlyMLMHead  
from transformers import BertConfig


class bioclinical_bert_decoder_architecture(pl.LightningModule):
    """
    Bioclinical BERT Decoder Architecture. 

    This class represents a decoder architecture based on the BioClinical BERT model. 
    This BERT model has been configured to be used as a decoder.

    
    Attributes:
        model (Hugging Face Transformer Model): The pre-trained BioClinical BERT model, configured as a decoder.
        reshape_to_hidden_size (nn.Linear): A linear layer used to reshape the encoded images to the correct dimensions, in case the encoded images are in the form of feature maps.
        img_embedding (nn.Embedding): An embedding layer used to reshape the encoded images to the correct dimensions, in case the encoded images are in the form of image tokens.

    Methods:
        forward(images, reports): Performs forward pass through the model.

    """    
    def __init__(self):
        super().__init__()
        
        # Load the pre-trained Bioclinical BERT model and configure it as decoder
        config = BertConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") 
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", is_decoder=True, add_cross_attention=True, hidden_size=HIDDEN_SIZE) 
        self.model.pooler = None
        self.model.cls = BertOnlyMLMHead(config) # Add the masked/causal language modelling head to the model to make this model trainable for autoregressive text generation.
        
        # To reshape the encoded images to the correct dimensions 
        if REPRESENTATION == 'feature_map':
            self.reshape_to_hidden_size = nn.Linear(IMAGE_SIZE[-1]*IMAGE_SIZE[-2], HIDDEN_SIZE).to('cuda')
        if REPRESENTATION == 'token':
            self.img_embedding = nn.Embedding(NR_IMG_TOKENS, HIDDEN_SIZE).to('cuda') 

      
    
    def forward(self, images, reports):

        # Process the encoded images to the correct dimensions
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

        # Make a mask for the reports to prevent attention on padding tokens
        reports_attention_mask = torch.where(reports != PAD_IDX, torch.tensor(1), torch.tensor(0))
        
        # Perform forward pass through the BERT model to get hidden states
        output_bert = self.model(input_ids=reports, encoder_hidden_states=images, encoder_attention_mask=None, attention_mask=reports_attention_mask) 
        output_bert = output_bert.last_hidden_state 
        
        # Pass the hidden states through the masked/causal language modelling head to get the prediction logits
        logits = self.model.cls(output_bert)
        
        #This model generates MAX_LENGTH predictions exluding the sos token. I defined MAX_LENGTH as including the SOS token. Hence I expect the model to generate MAX_LENGTH-1 predictions. Thus I remove the last prediction from the logits.
        logits = logits[:, :-1, :]
        
        return logits 
    

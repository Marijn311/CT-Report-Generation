import pytorch_lightning as pl
from config import *
from dataset import data_module
from autoregressive import *
from model import model
from pytorch_lightning.callbacks import ModelCheckpoint

# Set seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


if __name__ == '__main__':
    """
    This is the main script. This is the only .py file that should be ran.
    This scripts loads a PyTorch Lightning model and data module.
    
    Depending on the configurations in config.py it either fits, validates, or performs autoregressive predictions
    using loaded model and data.
    """

    # Load the data module and model
    data_module = data_module()
    model = model()

    # Callback to save model checkpoints    
    checkpoint_callback = ModelCheckpoint(
        dirpath= CKPT_PATH_TO_SAVE,  
        monitor='val_loss',  
        filename='model-{step}-{val_loss:.2f}-{val_accuracy:.2f}',  
        save_top_k=3,  
        mode='min',  
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        num_sanity_val_steps=0,
        accumulate_grad_batches=GRAD_ACCUM,
        max_epochs=NUM_EPOCHS, 
        precision=PRECISION,
        log_every_n_steps=20,
        #val_check_interval=0.2,
        callbacks=[checkpoint_callback],        
        ) 

    # Either fit the model, validate the model, or perform autoregressive report generation
    if STAGE == 'fit':
        trainer.fit(model, data_module)
         
    if STAGE == 'validate':
        checkpoint = torch.load(CKPT_PATH_TO_LOAD)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        trainer.validate(model, data_module)
    
    if STAGE == 'autoregressive':
        checkpoint = torch.load(CKPT_PATH_TO_LOAD)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        autoregressive_object = autoregressive(model, data_module)
        autoregressive_object.autoregressive_inference()
        
        
         
              
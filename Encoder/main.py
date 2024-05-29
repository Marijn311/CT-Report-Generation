import pytorch_lightning as pl
from config import *
from dataset import data_module
from model import model
from pytorch_lightning.callbacks import ModelCheckpoint

#Set seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

"""
This is the main script. This is the only .py file that should be ran.
This scripts loads a PyTorch Lightning model and data module. 
Depending on the configurations in config.py it either fits, validates, or performs testing using loaded model and data.
"""
    
if __name__ == '__main__':
    
    # Load the data module and model
    data_module = data_module()
    model = model()
    
    
    # Callback to save model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPT_PATH_TO_SAVE,
        monitor='val_loss',  
        filename='model-{step}-{val_loss:.2f}-{macro_avg_optimal_accuracy}', 
        save_top_k=3,  
        mode='min',
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator=ACCELERATOR, 
        num_sanity_val_steps=0, # This has to be 0. In this project the validation_epoch_end function calls the training_epoch_end function. This will give an error for sanity val steps because the there is no data from the training epoch.
        accumulate_grad_batches=GRAD_ACCUM, 
        devices=DEVICES,
        max_epochs=NUM_EPOCHS, 
        precision=PRECISION, 
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        ) 
    
    
    # Either fit, validate, or test the model
    if STAGE == "fit":
        if SAVE_ENCODED_IMAGES == False:
            trainer.fit(model, data_module)
        else:
            checkpoint = torch.load(CKPT_PATH_TO_LOAD)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.validate(model, data_module)
   
    if STAGE == "validate":
        checkpoint = torch.load(CKPT_PATH_TO_LOAD)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.validate(model, data_module)
    
    if STAGE == "test":
        checkpoint = torch.load(CKPT_PATH_TO_LOAD)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.test(model, data_module)
        
        

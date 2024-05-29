import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import random_split
import numpy as np
import pandas as pd
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import *
import torch.utils.data
import matplotlib.pyplot as plt
import pickle

class dataset(torch.utils.data.Dataset):
    """
    A dataset class for encoded images and report pairs.

    This class represents a dataset that contains encoded images and their corresponding report pairs.
    It inherits from the `torch.utils.data.Dataset` class.

    Attributes:
        global_len (int): The total number of image-label pairs in the dataset.
        hospital_dataset_lengths (list): A list that stores the number of patients per hospital.
        report_corpus (list): A list that contains the cleaned report corpus.
        tokenized_reports (numpy.ndarray or torch.Tensor): The tokenized reports.

    Methods:
        __init__(): Initializes the dataset, loads and preprocesses the reports.
        __len__(): Returns the total number of image-label pairs.
        get_position(global_index, dataset_sizes): Converts a global index to a specific dataset and an index in that dataset.
        __getitem__(index): Returns one encoded image and one tokenized report.
    """

    def __init__(self):
        super().__init__()

        # Initialize attributes of the dataset class
        self.global_len = 0
        self.hospital_dataset_lengths = []
        self.report_corpus = []
        self.tokenized_reports = None

        # Used in intermediate step during processing
        report_pseudo_ids = []
        image_pseudo_ids = []
        report_corpus = []
        reports_corpus_processed = []

        #Load the reports. Count the number of reports and the number of images, make sure this count is the same.
        for hospital_name in HOSPITAL_NAMES:
            # Load the labels file
            labels_path = os.path.join(IMAGE_DIR, hospital_name, 'labels.xlsx')
            labels = pd.read_excel(labels_path)

            # Count the labels (by extracting all PseudoIDs)
            nr_labels = len(labels.index)  
            self.hospital_dataset_lengths.append((hospital_name, nr_labels))
            self.global_len += nr_labels
            report_ids = labels['PseudoID'].tolist()  
            report_pseudo_ids.extend(report_ids)

            # Count the images (by extracting all PseudoIDs)
            for _, folders, _ in os.walk(os.path.join(IMAGE_DIR, hospital_name)):
                for folder in folders:
                    image_pseudo_ids.append(folder)

            # Load the reports from the labels file
            report_text = labels['Report'].tolist()
            report_corpus.extend(report_text)
        assert len(report_corpus) == self.global_len



        # Preprocess the reports 
        for report in report_corpus:
            report = report.strip()                                                # Strip ALL trailing whitespaces
            report = re.sub(r"([.,!?])", r" \1 ", report)                          # Add a whitespace before and after each punctuation mark
            if ARCHITECTURE == 'transformer_scratch':
                report = '<SOS> ' + report + ' <EOS>'                              # Add <SOS> and <EOS> tokens to each report because the tokenizer that is used in the scratch-model doesnt do this automatically when tokenizing th reports
            reports_corpus_processed.append(report)                                # Add the cleaned report to the cleaned report corpus
            self.report_corpus = reports_corpus_processed                          # Overwrite the old report corpus with the processed report corpus
        print(f"Report corpus contains {len(self.report_corpus)} reports")
        
        
        
        # Tokenize the reports
        if ARCHITECTURE == 'transformer_scratch':
            # Fit tokenizer on the report corpus. Extract the vocab dict
            TOKENIZER.fit_on_texts(self.report_corpus)
            VOCAB_DICT = TOKENIZER.word_index
            VOCAB_DICT['<PAD>'] = 0  # Add pad token to the vocab dict, because the tokenizer for the scratch-model does not do this automatically
            
            # Check some assumptions
            print(f"Found {len(VOCAB_DICT)} unique tokens in the report corpus")
            assert len(VOCAB_DICT) == VOCAB_SIZE, f"you need to update the VOCAB_SIZE in the config file to {len(VOCAB_DICT)}"
            assert VOCAB_DICT['sos'] == SOS_IDX, f"you need to update the SOS_IDX in the config file to {VOCAB_DICT['sos']}"
            assert VOCAB_DICT['eos'] == EOS_IDX, f"you need to update the EOS_IDX in the config file to {VOCAB_DICT['eos']}"
          
            # Save the vocab dict to a pickle file. This saved fileis required during initialization of the model.
            with open(f'vocab_dict_scratch_{TASK}.pkl', 'wb') as f:
                pickle.dump(VOCAB_DICT, f)
            
            # Tokenize the reports with the fitted tokenizer. Pad and truncate the reports to the max length
            unpadded_untruncated_tokenized_reports = TOKENIZER.texts_to_sequences(self.report_corpus)
            self.tokenized_reports = pad_sequences(unpadded_untruncated_tokenized_reports, maxlen=MAX_LENGTH, padding='post', truncating='post')


        # Tokenize the reports
        if ARCHITECTURE == 'bioclinical_bert' or ARCHITECTURE == 'bio_gpt':
            # Tokenize the reports with the pretrained tokenizer. Pad and truncate the reports to the max length
            self.tokenized_reports = TOKENIZER(self.report_corpus, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="pt")["input_ids"]
            
            # Tokenize the reports with the pretrained tokenizer. This time without padding and truncation. This is used for plotting the original report lengths, which is useful for setting the max length
            unpadded_untruncated_tokenized_reports = []
            for report in self.report_corpus:
                unpadded_untruncated_tokenized_report = TOKENIZER(report, padding=False, truncation=False, return_tensors="pt")["input_ids"]
                unpadded_untruncated_tokenized_report = unpadded_untruncated_tokenized_report[0].tolist()
                unpadded_untruncated_tokenized_reports.append(unpadded_untruncated_tokenized_report)
                

        # Make a histogram of the report lengths in tokens. Useful for setting the max length
        lengths = [len(arr) for arr in unpadded_untruncated_tokenized_reports]
        if PLOT_REPORT_LENGTHS:
            plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), align='left', rwidth=0.8)
            plt.xlabel('Report Length (tokens)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Report Length. N={len(lengths)}')
            plt.show()
        print(f"Longest report is {max(lengths)} tokens long. MAX_LENGTH for the predictor is set to {MAX_LENGTH}. Reports will be truncated or padded to this length.")


    def __len__(self):
        """Returns the total number of image-label pairs."""
        return self.global_len

    def get_position(self, global_index, dataset_sizes):
        """Converts a global index to a specific dataset and an index in that dataset."""
        current_index = global_index
        for count, dataset_size in enumerate(dataset_sizes):
            if current_index < dataset_size:
                return count, current_index
            current_index -= dataset_size

    def __getitem__(self, index):
        """Returns one encoded image and its tokenized report."""
        # Convert the index to a specific patient in a specific hospital
        dataset_sizes = [dataset_length[1] for dataset_length in self.hospital_dataset_lengths]
        dataset_index, position_in_dataset = self.get_position(index, dataset_sizes)
        hospital_name = self.hospital_dataset_lengths[dataset_index][0]
        dataset_folder_path = os.path.join(IMAGE_DIR, hospital_name)
        subfolders = [f.path for f in os.scandir(dataset_folder_path) if f.is_dir()]
        subfolder = subfolders[position_in_dataset]
        subfolder_name = os.path.basename(subfolder)
      
        # Load the encoded image
        enc_img_path = os.path.join(IMAGE_DIR, hospital_name, subfolder_name, IMAGE_FILE_NAME)
        image_array = torch.load(enc_img_path)

        # Load the tokenized report associated with the image
        tokenized_report = self.tokenized_reports[index]
        
        # Depending on which tokenizer is used, the tokenized report is either a numpy array or a tensor. Convert it to a tensor if it is a numpy array
        if isinstance(tokenized_report, np.ndarray):
            tokenized_report = torch.from_numpy(tokenized_report)
        else:
            tokenized_report = tokenized_report.clone().detach()

        # Print an example of the input reports and input encoded image in the form that they will be given to the dataloaders
        if SHOW_DATA_EXAMPLES:
            # Make a dictionary that maps the tokenized report to the words
            token_to_word = dict([(value, key) for (key, value) in VOCAB_DICT.items()])
            
            # Untokenize the report and print it 
            tokenized_report_array = tokenized_report.numpy() if hasattr(tokenized_report, 'numpy') else tokenized_report.cpu().numpy()
            untokenized_report = [token_to_word.get(i, '?') for i in tokenized_report_array]
            untokenized_report = ' '.join(untokenized_report)
            print("\n\nNOW SHOWING AN EXAMPLE OF THE REPORTS AND ENCODED IMAGE, AS THEY ARE RETURNED BY THE DATALOADERS")
            print(f"\nReport: {untokenized_report}")

            # Print the encoded image array and its min and max values
            print(f"\nEncoded image array is: {image_array}")
            print(f"\nMin value of the image: {torch.min(image_array)}")
            print(f"\nMax value of the image: {torch.max(image_array)}")
            print(f"\nThis max value helps to determine the MAX_ENC_IMG_VALUE in the config file")

        return image_array, tokenized_report



class data_module(pl.LightningDataModule):
    """
    LightningDataModule subclass for handling the data module of the model.

    This class provides the necessary methods to set up the data module, including splitting the dataset into train, validation, and test sets,
    and creating dataloaders for each set.

    Args:
        None

    Attributes:
        entire_ds (Dataset): The entire dataset.
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        test_ds (Dataset): The test dataset.

    Methods:
        setup(stage): Method to set up the data module.
        train_dataloader(): Method to create a dataloader for the training dataset.
        val_dataloader(): Method to create a dataloader for the validation dataset.
        test_dataloader(): Method to create a dataloader for the test dataset.
    """
    def __init__(self):
        super().__init__()
       
    def setup(self, stage):
        self.entire_ds = dataset() 
        generator1 = torch.Generator().manual_seed(42) # To ensure reproducibility in train/val/test split
        self.train_ds, self.val_ds, self.test_ds = random_split(self.entire_ds, DATA_SPLIT, generator=generator1)
      
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=True)
    

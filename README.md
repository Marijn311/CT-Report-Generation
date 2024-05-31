# CT-To-Report

## Description
This is the Python implementation of the work presented in "Automatically Generating Narrative-Style Radiology Reports from Volumetric CT Images; a Proof of Concept", available on [ArXiv](todo: include hyperlink) and this repository.

This repository is split into three main sections: <br>
* "Encoder" contains a Pytorch Lightning framework to train a (pseudo) 3D CNN to perform multi-label classification on volumetric CT data.<br>
* "Decoder" contains a Pytorch Lightning framework to train a LLM to generate narrative-style radiology reports, conditioned on the encoded images.<br>
* "SARLE" contains an algorithm to automatically mine classification labels from radiology reports [1].<br>

If you find this work useful in your research, please consider citing me: (Todo include bibtex).<br>
If you have any question you can reach me at m.m.borghouts@student.tue.nl (no promises or guanrantees).<br>


## Usage
1 -- This project has been designed to work on a single high-end personal graphics card with 12GB of video RAM. 
As a result, some concessions have been made that negatively impact overall performance.
Most importantly, the encoder and decoder are trained seperately. Therefore, these components should be treated as separate projects.
For example, when working with the encoder, you should open the "Encoder" folder in the IDE workspace instead of the main folder, as both the encoder and decoder have their own config files with overlapping variable names.
If you have more computational resources available you can make several improvements to this project. For this, see the future work suggestions in the corresponding paper.      


2 -- Though the code and methodology for this work is publicly available, the dataset is not. 
Hence, the first step to using this framework is to obtain a dataset of volumetric CT images with associated narrative-style radiology reports.
The dataset should be formatted as:

[main image dir]<br>
│   [data folder_1]<br>
│   │   [data folder_1]_[patient_id_1]<br>
│   │   │   [image file name]<br>
│   │   [data folder_1]_[patient_id_2]<br>
│   │   │   [image file name]<br>
│   │   ...<br>
│   │   labels.xlsx (created by SARLE from reports.docx)<br>
│   │   reports.docx<br>
│   [data folder_2]<br>
│   │   [data folder_2]_[patient_id_1]<br>
│   │   │   [image file name]<br>
│   │   [data folder_2]_[patient_id_2]<br>
│   │   │   [image file name]<br>
│   │   ...<br>
│   │   labels.xlsx (created by SARLE from reports.docx)<br>
│   │   reports.docx<br>
│   ...<br>
│   all_labels.xlsx (created by SARLE from reports.docx)<br>
│   all_reports.docx (created by SARLE from reports.docx)<br>
│   SARLE_dataset.xlsx (created by SARLE from reports.docx)<br>

The benefit of this seemingly awkward file structure is that individual data folders can be added or removed at any time. 
This is because patient IDs can restart from 1 for each data folder, allowing IDs to overlap as long as patients are in different folders. 
Additionally, patient IDs do not need to be continuous. For example, if patient ID 5 is removed due to image quality, there is no need to renumber all subsequent patients to fill the gap.


3 -- Next, the images and radiology reports need to be pre-processed. The pre-processing of the reports should remove all information that the encoder is not expected to capture, such as dates, names of patients/staff/hospitals, and references to previous imaging studies.
For more details and suggestions, see the "Methodology: Dataset" section in the corresponding paper.
There are two stages of image pre-processing. Firstly, images may be pre-processed in ways that permanently alter the image files.
Scripts for this are available in the Encoder/dataset_preprocessing directory, which modify all images in the main image directory using a large loop. Options include:

a) Creating segmentation masks for up to 104 anatomical regions using the TotalSegmentor models [2].<br>
b) Applying segmentation masks to the images.<br>
c) Introducing individual or multiple artificial abnormalities to the images, such as mirroring, rotation, or occlusion of anatomical regions.<br>

Option c) exists to create surrogate classification tasks that are 1) visually easy to detect, 2) evenly distributed among classes, and 3) allow for an effective 5x increase in dataset size. 
This approach is useful if the available dataset is small with infrequent classes. For more details, refer to the "Methodology: Surrogate Tasks" section in the paper.

The second form of image pre-processing is done in the model's dataloader and does not affect the original image files. This pre-processing is definded in Encoder/dataset_preprocessing/ct_vol_preprocessing.py and includes things like image rescaling and normalization.

4 -- Run the SARLE algorithm to generate classification labels from the radiology reports. These classification labels are used to train the encoder. To do this, run SARLE/create_sarle_dataset.py. This script converts all the reports.docx files into one large Excel file, which can then be used by SARLE/main.py to extract the classification labels. The label extraction can be customized by modifying the medical vocabulary in SARLE/abnormality_vocabulary.py and the rules in SARLE/rules.py. For more details, refer to the docstrings in SARLE, the related paper, and the paper that introduced SARLE [1].

5 -- Train an encoder model using Encoder/main.py, Encoder/config.py, the volumetric images, and the classification labels generated by SARLE. Various models are available, and more can be added in Encoder/Model_architectures. Metrics such as binary cross-entropy, classification accuracy, and the precision-recall curve are available.

6 -- Pass all images through a trained encoder model and save the encoded images as .pt files. To do this, set SAVE_ENCODED_IMAGES to True in Encoder/config.py and follow the instructions in Encoder/utils.py.

7 -- Train a decoder model using Decoder/main.py, Decoder/config.py, the encoded images in .pt form, and the radiology reports. Various models are available, and more can be added in Decoder/Model_architectures. Metrics such as binary cross-entropy and next word prediction accuracy are available.

8 -- Autoregressively generate radiology reports using Decoder/autoregressive.py, Decoder/config.py, the encoded images in .pt form, and the radiology reports. Metrics such as next word prediction accuracy and NLP metrics like BLEU, METEOR, and ROUGE are available.

## References
[1] Draelos, Rachel Lea, David Dov, Maciej A. Mazurowski, Joseph Y. Lo, Ricardo Henao, Geoffrey D. Rubin, and Lawrence Carin. "Machine-learning-based multiple abnormality prediction with large-scale chest computed tomography volumes." Medical Image Analysis 67 (2021): 101857.<br>

[2] Jakob Wasserthal, Hanns-Christian Breit, Manfred T. Meyer, Maurice Pradella, Daniel Hinck, Alexander W. Sauter, Tobias Heye, Daniel T. Boll, Joshy Cyriac, Shan Yang, Michael Bach, and Martin Segeroth. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images". Radiology: Artificial Intelligence 2023 5:5.<br>

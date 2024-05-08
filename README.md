# CT-To-Report

# Description

This is the Python implementation of the work presented in "Recognise and Describe: Automatically Generating Narrative-Style Radiology Reports from Volumetric CT Images, a Proof of Concept", available on ArXiv (todo: include hyperlink).

This repository is split into 3 main sections: Encoder decoder and SARLE. 
The Encoder contains a Pytorch Lightning framework to train a multilabel classification model on volumetric data.
The Decoder contains a Pytorch Lightning framework to train a LLM to generate radiology reports which are conditioned in the encoded images that are outputted by the decoder
The SARLE section contains code to automatically mine classification labels from radiology reports.


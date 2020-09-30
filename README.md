## Description
Repository of Deep Propensity Network - Sparse Autoencoder (DPN-SA) to calculate propensity score using sparse autoencoder.

## Requirements and setup
pytorch - 1.3.1 <br/>
numpy - 1.17.2 <br/>
pandas - 0.25.1 <br/>
scikit - 0.21.3 <br/>
matplotlib: 3.1.1 <br/>
python -  3.7.4 <br/>


## Keywords
causal AI, biomedical informatics, deep learning, multitask learning, sparse autoencoder

## Dependencies
[python 3.7.7](https://www.python.org/downloads/release/python-374/)

[pytorch 1.3.1](https://pytorch.org/get-started/previous-versions/)


## How to run
To reproduce the experiments mentioned in the paper for DCN-PD, Logistic regression, Logistic regression Lasso 
and all the SAE variants of 25-20-10- (greedy and end to end) for both the
original and synthetic dataset, type the following
command: 

<b>python3 main_propensity_dropout.py</b>

## Output
After the run, the outputs will be generated in the following location:

<b>[IHDP](https://github.com/Shantanu48114860/DPN-SA/tree/master/MSE) </b>

Consolidated results will be available in textfile in /Details_original.txt and /Details_augmented.txt files.

The details of each run will be avalable in csv files in the following locations:

/MSE/Results_consolidated.csv and /MSE_Augmented/Results_consolidated_NN.csv



## License & copyright
© DISL, University of Florida

Licensed under the [MIT License](LICENSE)

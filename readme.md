# Liver Cirrhosis Prediction Project
## Project Overview
This is the final project of EN.601.475/675 focusing on predicting liver cirrhosis using machine learning techniques. The repository contains various components for data analysis, model training, and evaluation.

## File Structure
* `EDA.ipynb`: Jupyter notebook containing exploratory data analysis
* `ML_Final_Proposal_1_.pdf`: Project proposal document
* `RF_KNN_SVM.ipynb`: Implementation of Random Forest model for cirrhosis prediction
* `mlp.ipynb`: Neural network implementation
* `pre_missing_data.ipynb`: Preprocessing notebook for handling missing data and generate dataset.
* `train_em.csv`: Training dataset using EM algorithm to process missing data.
* `train_mix.csv`: Training dataset using median/mode imputation with noise for the data of 106 patients. The rest missing data would be processed by EM algorithm.

## Getting Started
To run this project:
1. Start with `EDA.ipynb` to understand the dataset and view the exploratory data analysis.
2. Run `pre_missing_data.ipynb` to handle missing values and create processed training datasets.
3. Open `RF_KNN_SVM.ipynb` to train and evaluate the Random Forest model.
4. Explore `mlp.ipynb` for training and evaluating the Neural Network model implementation.
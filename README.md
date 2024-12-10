# Comapraing Different ML methods in predicting Breast Cancer Pathology

This project focuses on predicting breast cancer using the Wisconsin Breast Cancer dataset. The dataset consists of 569 samples, which are derived from Fine Needle Aspiration Cytology (FNAC) images of breast masses (biopsies). The features extracted from these images were computed into numerical values, providing the dataset with a total of 32 features. Among these features, the dataset includes:

10 core features: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (with the mean values for each of these features).
The standard error and worst (largest) values of these same features.
Additional fields for diagnosis (Malignant or Benign) and a unique ID for each sample.

## Our Goal
The primary objective is to compare  four different machine learning models and their performance in predicting whether a tumor is malignant or benign. We also aim to optimize these models using feature importance techniques to improve their prediction accuracy.
1. Support Vector Machine (SVM)
2. K-nearnest Neighbors (KNNs)
3. Decision Trees
4.  Logistic Regression 

## Exploratory Data Analysis (EDA)
Initial exploratory data analysis has revealed notable differences between benign and malignant cell features. These differences may prove to be significant in improving the accuracy of our prediction models, and further investigation into these patterns will guide feature selection and model optimization.


## Setup 
Step 1: create venv On terminal 
```python3 -m venv venv```

Step 2 : activate venv
```source venv/bin/activate```

Step 3 :  Install necessay libraries
- dependencies on requirements.txt
``` pip install -r requirements.txt``` 

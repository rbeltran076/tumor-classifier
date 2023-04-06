# This machine learning model will predict cancer from any given brain tumor
# physical characteristics.

# Import the necessary libraries for ML
# I am going to use sk.learn in this model
# But other libraries also are designed for ML, like Tensorflow and Pytorch

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# First read of the tumor dataset
unfilteredTumorData = pd.read_csv("brain tumors model/tumordata.csv")

# Rewrite the table with no empty spaces so that everything is used.
tumorData = unfilteredTumorData.dropna(axis = 0)
print(f"The dataset is now filtered, it is of shape {tumorData.shape}")

# Define the diagnosis, the prediction, as the Y item. The output.
# Similarly, the rest of the items as X. The inputs.
yItem = tumorData.Diagnosis
xItem = tumorData[[
    "Radius (mean)",
    "Texture (mean)",
    "Perimeter (mean)",
    "Area (mean)", 
    "Smoothness (mean)",
    "Compactness (mean)",
    "Concavity (mean)",
    "Concave points (mean)",
    "Symmetry (mean)",
    "Fractal dimension (mean)",
    "Radius (se)",
    "Texture (se)",
    "Perimeter (se)",
    "Area (se)",
    "Smoothness (se)",
    "Compactness (se)",
    "Concavity (se)",
    "Concave points (se)",
    "Symmetry (se)",
    "Fractal dimension (se)",
    "Radius (worst)",
    "Texture (worst)",
    "Perimeter (worst)",
    "Area (worst)",
    "Smoothness (worst)",
    "Compactness (worst)",
    "Concavity (worst)",
    "Concave points (worst)",
    "Symmetry (worst)",
    "Fractal dimension (worst)"
]]


# The model itself. I will use a technique named Random Forest Regression.
# Random Forest Regression is a scaled up version of a decision tree, although
# I am not sure of that. And I do not know how either of those actually work.
# -a description is given in the pop up documentation anywat
diagnosisModel = RandomForestRegressor()

# Create train and test data for building the model.
dataTestingPercentage = 0.75
xTrain, xTest, yTrain, yTest = train_test_split(
    xItem, yItem,                        # use the items declared above as input and output respectively
    test_size = dataTestingPercentage,   # test with 0.25 of the total cases
    random_state = None)                   # the random seed

# fit the model with the test data
diagnosisModel.fit(xTest, yTest)
print(f"There is now a model for the diagnosis from {dataTestingPercentage} of this dataset")

# Make some predictions
predictions = diagnosisModel.predict(xTest)

# use the mean absolute error to know how accurate the model is
mae = mean_absolute_error(yTest, predictions)
print(f"""
The validation is finished.
the mae of this model is {mae}
This model has an accuracy of {1 - mae}
""")
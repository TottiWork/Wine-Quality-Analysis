#Panggil file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# me-non aktifkan peringatan pada python
import warnings
warnings.filterwarnings('ignore')

dataset = "/content/data_training.csv"
df = pd.read_csv(dataset)

print(df.info())

print("Summary of training data:")
print(df.describe())

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def load_data():
    df = pd.read_csv("data/dataset-diabete.csv")
    print("Data loaded successfully")
    return df


def clean_data(df):
    df = df.drop_duplicates()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def impute_missing_values(df):
    cols_with_fake_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    df[cols_with_fake_zeros] = df[cols_with_fake_zeros].replace(0, np.nan)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    knn = KNNImputer(n_neighbors=5)
    filled = knn.fit_transform(scaled)

    df = pd.DataFrame(scaler.inverse_transform(filled), columns=df.columns)
    return df


def treat_outliers(df):
    df["BloodPressure"] = df["BloodPressure"].clip(lower=40, upper=120)
    df["BMI"] = df["BMI"].clip(lower=15, upper=50)
    df["Glucose"] = df["Glucose"].clip(lower=40, upper=300)
    df["SkinThickness"] = df["SkinThickness"].clip(lower=5, upper=60)
    df["Insulin"] = df["Insulin"].clip(lower=15, upper=550)
    return df


def preprocess_data(df):
    df = clean_data(df)
    df = impute_missing_values(df)
    df = treat_outliers(df)
    return df
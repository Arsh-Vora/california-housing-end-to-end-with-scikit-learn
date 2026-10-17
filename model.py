"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
import pathlib
import tarfile
import tempfile
import urllib.request
import pandas as pd

def load_housing() -> pd.DataFrame:
    tar_path = pathlib.Path(tempfile.gettempdir()) / "housing.tgz"

    if not tar_path.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tar_path)

    with tarfile.open(tar_path) as tar:
        csv_file = tar.extractfile("housing/housing.csv")
        df = pd.read_csv(csv_file)

    return df

# Step 2 - income_categories
def income_categories(df):
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    return pd.cut(df["median_income"], 
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
    labels=[1, 2, 3, 4, 5]).astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state=42):
    strata = income_categories(df)
    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strata
    )
    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    corr_matrix = df.corr(numeric_only=True)
    
    target_corr = (
        corr_matrix["median_house_value"]
        .drop("median_house_value")
        .sort_values(ascending=False)
    )
    
    return target_corr

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    result = df.copy()

    result["rooms_per_house"] = result["total_rooms"] / result["households"]
    result["bedrooms_ratio"] = result["total_bedrooms"] / result["total_rooms"]
    result["people_per_house"] = result["population"] / result["households"]

    return result

# Step 6 - split_features_labels
def split_features_labels(df):
    # TODO: Return (X without median_house_value, y = median_house_value Series).
    X =df.drop("median_house_value", axis=1)
    y=df["median_house_value"].copy()

    return X,y

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # TODO: store the parameters
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # TODO: fit KMeans(n_clusters, n_init=10, random_state) on X with sample_weight; keep it as self.kmeans_
        self.kmeans_ = KMeans(
        n_clusters=self.n_clusters, 
        n_init=10, 
        random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self


    def transform(self, X):
        # TODO: rbf_kernel similarity of each row of X to the cluster centers
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        # TODO: ["Cluster 0 similarity", ...]
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Step 8 - numeric_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

# Step 9 - categorical_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(missing_values=None, strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

# Step 10 - build_preprocessing
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )
    
    log_cols = ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    
    return ColumnTransformer(
        transformers=[
            ("log", log_pipeline, log_cols),
            ("geo", ClusterSimilarity(n_clusters=n_clusters, gamma=gamma, random_state=random_state), ["latitude", "longitude"]),
            ("cat", categorical_pipeline(), ["ocean_proximity"])
        ],
        remainder=numeric_pipeline()
    )

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement


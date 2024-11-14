import numpy as np
from sklearn.preprocessing import FunctionTransformer, \
    OrdinalEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from . import helper_functions

# Define the preprocessing components
log_transformer = FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")
std_scaler = StandardScaler()
one_hot = OneHotEncoder()
mode_imputer = SimpleImputer(strategy="most_frequent")
mean_imputer = SimpleImputer(strategy="mean")

weekday_order = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
education_order = ["Lower secondary", "Secondary / secondary special",
                   "Incomplete higher", "Higher education",
                   "Academic degree"]

ord_encoder_day = OrdinalEncoder(categories=[weekday_order])
ord_encoder_edu = OrdinalEncoder(categories=[education_order])

ord_encoder = OrdinalEncoder()
ordinal_transformer = Pipeline([
    ("encoder", OrdinalEncoder()),
    ("imputer", mean_imputer)
])

log_robust = Pipeline([
    ("mean_imp", mean_imputer),
    ("log", log_transformer),
    ("robust", RobustScaler())
])

std = Pipeline(steps=[
    ("mean_imp", mean_imputer),
    ("std", std_scaler)
])

log_pipe = Pipeline([
    ("imputer", mean_imputer),
    ("log", log_transformer)
])

def baseline_preprocessing_pipeline(log_rob_cols: List[str],
                                    std_cols: List[str],
                                    one_hot_cols: List[str],
                                    ext_source_cols: List[str],
                                    passthrough_cols: List[str])  -> ColumnTransformer:
    """
    Creates a preprocessing pipeline for data transformation, including
    standardization, logarithmic transformations with robust scaling,
    one-hot encoding, ordinal encoding, and missing value imputation.

    Parameters:
    -----------
    log_rob_cols : List[str]
        List of column names for logarithmic transformation and robust scaling.
    std_cols : List[str]
        List of column names for standardization (mean-imputation followed by standard scaling).
    one_hot_cols : List[str]
        List of column names for one-hot encoding.
    ext_source_cols : List[str]
        List of column names for mean imputation (e.g., external source data).
    passthrough_cols : List[str]
        List of column names to be passed through without any transformations.

    Returns:
    --------
    preprocessing : ColumnTransformer
        A ColumnTransformer object that applies the specified transformations to the data.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("std", std, std_cols),
            ("log_robust", log_robust, log_rob_cols),
            ("one_hot", one_hot, one_hot_cols),
            ("weekday_ord", ord_encoder_day, ["WEEKDAY_APPR_PROCESS_START"]),
            ("edu_ord", ord_encoder_edu, ["NAME_EDUCATION_TYPE"]),
            ("gender_ord", ord_encoder, ["CODE_GENDER", "NAME_CONTRACT_TYPE"]),
            ("ext_source", mean_imputer, ext_source_cols),
            ("passthrough", "passthrough", passthrough_cols)
        ])

    return preprocessing

def updated_preprocessing_pipeline(log_rob_cols: List[str],
                                    std_cols: List[str],
                                   log_cols: List[str],
                                    one_hot_cols: List[str],
                                   ordinal_encoding_cols: List[str],
                                    ext_source_cols: List[str],
                                    passthrough_cols: List[str])  -> ColumnTransformer:
    """
    Creates a preprocessing pipeline for data transformation, including
    standardization, logarithmic transformations with robust scaling,
    one-hot encoding, ordinal encoding, and missing value imputation.

    Parameters:
    -----------
    log_rob_cols : List[str]
        List of column names for logarithmic transformation and robust scaling.
    std_cols : List[str]
        List of column names for standardization (mean-imputation followed by standard scaling).
    log_cols : List[str]
        List of column names for logarithmic transformation.
    one_hot_cols : List[str]
        List of column names for one-hot encoding.
    one_hot_cols : List[str]
        List of column names for ordinal encoding pipeline.
    ext_source_cols : List[str]
        List of column names for mean imputation (e.g., external source data).
    passthrough_cols : List[str]
        List of column names to be passed through without any transformations.

    Returns:
    --------
    preprocessing : ColumnTransformer
        A ColumnTransformer object that applies the specified transformations to the data.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            ("std", std, std_cols),
            ("log_robust", log_robust, log_rob_cols),
            ("log", log_pipe, log_cols),
            ("one_hot", one_hot, one_hot_cols),
            ("weekday_ord", ord_encoder_day, ["WEEKDAY_APPR_PROCESS_START"]),
            ("edu_ord", ord_encoder_edu, ["NAME_EDUCATION_TYPE"]),
            ("ordinal_encoder", ordinal_transformer, ordinal_encoding_cols),
            ("mean_imputer", mean_imputer, ext_source_cols),
            ("passthrough", "passthrough", passthrough_cols)
        ])

    return preprocessing

def encode_features_for_correlation(
    data: pd.DataFrame,
    one_hot_encode_cols: List[str],
    ordinal_encoder: OrdinalEncoder = ord_encoder,
    ordinal_encoder_edu: OrdinalEncoder = ord_encoder_edu,
    ordinal_encoder_day: OrdinalEncoder = ord_encoder_day,
) -> pd.DataFrame:
    """
    Encodes features in a dataset for numeric analysis, especially for correlation calculations.
    Handles categorical encoding, one-hot encoding, and filling missing values.

    Parameters:
    - data: DataFrame with features to encode.
    - ordinal_encoder: OrdinalEncoder instance for encoding gender-related columns.
    - ordinal_encoder_edu: OrdinalEncoder instance for encoding education level.
    - ordinal_encoder_day: OrdinalEncoder instance for encoding weekday order.
    - one_hot_encode_cols: List of columns to apply one-hot encoding to.

    Returns:
    - DataFrame with all specified columns encoded for correlation analysis.
    """
    encoded_data = data.copy()

    float_columns = encoded_data.select_dtypes("float")
    encoded_data[float_columns.columns] = float_columns.fillna(float_columns.mean())

    encoded_data = pd.concat(
        [encoded_data, pd.get_dummies(encoded_data[one_hot_encode_cols])],
        axis=1
    )
    encoded_data.drop(columns=one_hot_encode_cols, inplace=True)

    label_encode_cols = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "WEEKDAY_APPR_PROCESS_START"]
    for col, encoder in zip(label_encode_cols,
                            [ordinal_encoder, ordinal_encoder_edu, ordinal_encoder_day]):
        encoded_data[[col]] = encoder.fit_transform(encoded_data[[col]])

    cat_cols = encoded_data.select_dtypes("object").columns
    encoded_data[cat_cols] = encoded_data[cat_cols].replace({"YES": 1, "NO": 0})
    encoded_data[cat_cols] = encoded_data[cat_cols].astype("float")

    encoded_data = encoded_data.fillna(encoded_data[cat_cols].mean())

    return encoded_data

def apply_smote(X: pd.DataFrame, y: pd.Series,
                sampling_strategy: float = 0.2, random_state: int = 0):
    """
    Applies SMOTE to increase the minority class samples.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target variable.
    - sampling_strategy (float): Ratio of minority to majority class samples.
    - random_state (int): Random state for reproducibility.

    Returns:
    - X_resampled (pd.DataFrame): Resampled features.
    - y_resampled (pd.Series): Resampled target.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_random_undersampling(X: pd.DataFrame, y: pd.Series,
                               sampling_strategy: float = 1.0, random_state: int = 0):
    """
    Applies Random Undersampling to reduce majority class samples.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target variable.
    - sampling_strategy (float): Ratio of minority to majority class samples.
    - random_state (int): Random state for reproducibility.

    Returns:
    - X_resampled (pd.DataFrame): Resampled features.
    - y_resampled (pd.Series): Resampled target.
    """
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                      random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_over_under_sampling(X: pd.DataFrame, y: pd.Series,
                              smote_strategy: float = 0.2,
                              under_strategy: float = 0.5,
                              random_state: int = 0):
    """
    Applies a combination of SMOTE and Random Undersampling.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target variable.
    - smote_strategy (float): SMOTE sampling ratio for the minority class.
    - under_strategy (float): Undersampling ratio after SMOTE.
    - random_state (int): Random state for reproducibility.

    Returns:
    - X_resampled (pd.DataFrame): Resampled features.
    - y_resampled (pd.Series): Resampled target.
    """
    over = SMOTE(sampling_strategy=smote_strategy, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
    over_under_pipeline = Pipeline([('smote', over), ('undersample', under)])
    X_resampled, y_resampled = over_under_pipeline.fit_resample(X, y)
    return X_resampled, y_resampled

def prepare_categorical_datasets(dataset,
                                 drop_column="SK_ID_CURR"):
    """
    Prepares dataset for models like LightGBM or CatBoost, which don't require
    categorical columns to be preprocessed.

    Parameters:
    - dataset: table that needs to be modified
    - drop_column (str): Column to drop from both datasets (e.g., 'SK_ID_CURR').

    Returns:
    - modified_dataset(pd.DataFrame): modified dataset
    """
    modified_dataset = dataset.drop(columns=drop_column)

    cat_columns = (list(modified_dataset.select_dtypes("category").columns) +
                   list(modified_dataset.select_dtypes("object").columns))

    modified_dataset[cat_columns] = modified_dataset[cat_columns].astype(str).fillna("UNKNOWN")
    modified_dataset[cat_columns] = modified_dataset[cat_columns].astype(str).fillna("UNKNOWN")

    helper_functions.reduce_memory_usage(modified_dataset)

    return modified_dataset



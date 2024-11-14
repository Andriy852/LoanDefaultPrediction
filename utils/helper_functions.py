import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import (accuracy_score, recall_score, 
                             precision_score, f1_score, 
                             confusion_matrix, roc_auc_score,
                             roc_curve)
import numpy as np
from sklearn.model_selection import cross_val_score
import math
import lightgbm as lgb
from catboost import CatBoostClassifier


def train_gradient_boosting_models(datasets, X_val_cat, y_val, cat_columns,
                                   lgbm_params, scale_pos_weight,
                                   num_boost_round=2000,
                                   early_stopping_rounds=100):
    """
    Trains and evaluates LightGBM and CatBoost models on multiple datasets.

    Parameters:
    - datasets (dict): A dictionary with dataset names as keys,
                        and tuples (X_train, y_train) as values.
    - X_val_cat (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation target.
    - cat_columns (list): List of categorical feature names for CatBoost.
    - lgbm_params (dict): Dictionary of LightGBM parameters.
    - scale_pos_weight (float): Weight to apply for LightGBM balancing.
    - num_boost_round (int): Number of boosting rounds for each model.
    - early_stopping_rounds (int): Early stopping rounds for both models.

    Returns:
    - roc_res (pd.DataFrame): DataFrame with AUC scores for
                            LightGBM and CatBoost models across datasets.
    """
    roc_res = pd.DataFrame(index=["LightGBM", "CatBoost"], columns=datasets.keys())

    for name, (X_train, y_train) in datasets.items():
        cat_boost = CatBoostClassifier(
            cat_features=cat_columns,
            random_seed=0,
            verbose=0,
            early_stopping_rounds=early_stopping_rounds,
            iterations=num_boost_round,
            eval_metric='AUC'
        )

        if name == "Balanced":
            cat_boost.set_params(auto_class_weights="Balanced")
            lgbm_params["scale_pos_weight"] = scale_pos_weight

        cat_boost.fit(X_train, y_train, eval_set=(X_val_cat, y_val))
        pred_probas_cat = cat_boost.predict_proba(X_val_cat)[:, 1]
        roc_res.loc["CatBoost", name] = roc_auc_score(y_val, pred_probas_cat)

        X_train_lgbm_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature="auto")
        X_val_lgbm_dataset = lgb.Dataset(X_val_cat, label=y_val, reference=X_train_lgbm_dataset)

        lgb_model = lgb.train(
            lgbm_params,
            X_train_lgbm_dataset,
            valid_sets=[X_train_lgbm_dataset, X_val_lgbm_dataset],
            valid_names=['train', 'val'],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds)
            ]
        )
        print(f"Dataset: {name}. CatBoost best iteration: {cat_boost.best_iteration_}")
        print(f"Dataset: {name}. LightGBM best iteration: {lgb_model.best_iteration}")

        roc_res.loc["LightGBM", name] = lgb_model.best_score["val"]["auc"]

    return roc_res


def plot_confusion_matrices(models: List[Dict],
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Plot confusion matrices for a list of models with specified training and validation datasets.

    Parameters:
    - models: List of dictionaries, where each dictionary contains:
        - "name": The name of the model (str)
        - "model": The model instance (BaseEstimator)
        - "X_train": Training data features (optional)
        - "y_train": Training labels (optional)
        - "X_val": Validation data features
        - "y_val": Validation labels
    - figsize: Size of the plot (width, height)

    Returns:
    - None
    """
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle("Confusion Matrices", fontsize=16, y=0.95)

    for i, model_info in enumerate(models):
        name = model_info["name"]
        model = model_info["model"]
        X_val = model_info["X_val"]
        y_val = model_info["y_val"]

        X_train = model_info.get("X_train")
        y_train = model_info.get("y_train")

        if X_train is not None and y_train is not None:
            model.fit(X_train, y_train)

        preds = np.round(model.predict(X_val))
        cm = confusion_matrix(y_val, preds)

        ax = fig.add_subplot(int(np.ceil(len(models) / 2)), 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name)

    plt.show()


def plot_roc_curves(models: List[dict]):
    """
    Plots ROC curves for a list of models.

    Parameters:
    - models: list of dictionaries with keys:
        - 'name': name of the model
        - 'model': model instance
        - 'X_val': validation features
        - 'y_val': true labels for the validation set
    - X_test: test features
    - y_test: true labels for the test set
    """
    plt.figure(figsize=(8, 6))

    for model_info in models:
        model_name = model_info["name"]
        model = model_info["model"]
        X_val = model_info["X_val"]
        y_val = model_info["y_val"]

        if isinstance(model, lgb.basic.Booster):
            y_pred_prob = model.predict(X_val)
        else:
            y_pred_prob = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
        auc = roc_auc_score(y_val, y_pred_prob)
        sns.lineplot(x=fpr, y=tpr, label=f'{model_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()


def plot_model_roc_auc(models: List[BaseEstimator],
                       datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plots ROC AUC scores of different models across various sampling techniques.

    Parameters:
    - models (list): List of scikit-learn models to evaluate.
    - datasets (dict): Dictionary of sampled datasets in the format
                       {'Dataset Name': (X_train, y_train)}.
    - X_val (pd.DataFrame): Validation feature data.
    - y_val (pd.Series): Validation target labels.
    """
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle("ROC AUC Scores of Different Models and Sampling Techniques",
                 fontsize=16, y=0.95)

    for i, model in enumerate(models):
        ax = fig.add_subplot(int(np.ceil(len(models) / 2)), 2, i + 1)
        roc_res = pd.Series(dtype=float) 

        for name, (X_train, y_train) in datasets.items():
            if name == "Balanced":
                model.set_params(class_weight="balanced")

            model.fit(X_train, y_train)
            pred_probas = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, pred_probas)
            roc_res[name] = roc_auc

        roc_res.plot(ax=ax, kind="bar", color="red")
        title = type(model).__name__
        ax.set_title(title, y=1.05)
        customize_bar(axes=ax, round_to=4, position="v")
        ax.set_ylabel("ROC AUC Score")
        ax.set_xlabel("Sampling Technique")

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces memory usage of a DataFrame by downcasting numerical columns
    and converting object columns to categorical where appropriate.

    Parameters:
        df (pd.DataFrame): The DataFrame to reduce memory usage for.

    Returns:
        pd.DataFrame: Optimized DataFrame with reduced memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Starting memory usage of dataframe: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')

        elif col_type == object:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Final memory usage of dataframe: {end_mem:.2f} MB")
    print(f"Reduced by: {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df

def get_scores(model: BaseEstimator, X: np.ndarray,
               y: np.ndarray, fit: bool = True) -> Dict[str, float]:
    """
    Compute performance scores on the data.

    Parameters:
    model (BaseEstimator): The machine learning model
    X (np.ndarray): The feature matrix used.
    y (np.ndarray): The target vector used.
    fit (bool): If True, the model will be fitted to the data. Default is True.

    Returns:
    Dict[str, float]: A dictionary containing accuracy, recall, precision, and f1 scores.
    """
    if fit:
        model.fit(X, y)

    model_predict = model.predict(X)
    model_predict_proba = model.predict_proba(X)[:, 1]

    scores = {
        "accuracy": accuracy_score(y, model_predict),
        "recall": recall_score(y, model_predict),
        "precision": precision_score(y, model_predict),
        "f1": f1_score(y, model_predict),
        "roc_auc": roc_auc_score(y, model_predict_proba)
    }

    return scores


def calculate_cross_val_scores(models: List,
                               X: pd.DataFrame,
                               y: pd.Series,
                               scoring: str = "roc_auc",
                               cv: int = 3) -> pd.DataFrame:
    """
    Calculate cross-validation scores for a list of models.

    Parameters:
    - models: List of (model_name, model) tuples.
    - X: Features data (DataFrame).
    - y: Target data (Series).
    - scoring: Scoring metric (default is "roc_auc").
    - cv: Number of cross-validation folds (default is 3).

    Returns:
    - DataFrame with model names and their corresponding AUC scores.
    """
    scores = {}

    for model_name, model in models:
        auc_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        scores[model_name] = auc_scores

    return pd.DataFrame(scores)

def customize_bar(position: str, axes, 
                  values_font=12, pct=False, round_to=0) -> None:
    """
    Function, which customizes bar chart
    Takes axes object and:
        - gets rid of spines
        - modifies ticks
        - adds value above each bar
    Parameters:
        - position(str): modify the bar depending on how the
        bars are positioned: vertically or horizontally
    Return: None
    """
    for spine in axes.spines.values():
        spine.set_visible(False)
    if position == "v":
        axes.set_yticks([])
        for tick in axes.get_xticklabels():
            tick.set_rotation(0)
    if position == "h":
        axes.set_xticks([])
        for tick in axes.get_yticklabels():
            tick.set_rotation(0)
    for bar in axes.patches:
        if bar.get_width() == 0:
            continue
        if position == "v":
            text_location = (bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 1/100*bar.get_height())
            value = bar.get_height()
            location = "center"
        elif position == "h":
            text_location = (bar.get_width(),
                             bar.get_y() + bar.get_height() / 2)
            value = bar.get_width()
            location = "left"
        if pct:
            value = f"{round(value * 100, round_to)}%"
        elif round_to == 0:
            value = str(int(value))
        else:
            value = str(round(value, round_to))
        axes.text(text_location[0],
                text_location[1],
                str(value),
                fontsize=values_font,
                ha=location)

def plot_cat_columns(data: pd.DataFrame, 
                     columns: List[str], 
                     title: str, 
                     figsize: Tuple[int, int], 
                     col_number: int = 3, 
                     rotate: int = 0, 
                     target: Optional[str] = None,
                    values_font: int = 10) -> None:
    """
    Plots count plots or bar plots for the specified categorical columns 
    in a DataFrame. If a target is provided, a bar plot will be used to show 
    the relationship between the categorical column and the target. 
    Otherwise, count plots will be generated.

    Parameters:
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    columns : List[str]
        A list of column names to be plotted. These columns should be categorical.
    title : str
        The title for the entire figure, displayed at the top of the plot.
    figsize : Tuple[int, int]
        The size of the figure (width, height) in inches.
    col_number : int, optional, default=3
        The number of plots to display per row. Determines the layout of subplots.
    rotate : int, optional, default=0
        The degree of rotation for the x-axis labels. Use this to adjust 
        label readability if they are too long.
    target : Optional[str], optional, default=None
        The name of the target column for plotting bar plots. If provided, 
        a bar plot showing the mean of the target for each category in the 
        column will be plotted. If not provided, a simple count plot is shown.

    Returns:
        None
    """
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle(title, fontsize=16, y=0.91)
    
    for i, column in enumerate(columns):
        ax = fig.add_subplot(len(columns) // col_number + 1, col_number, i + 1)
        if target:
            sns.barplot(x=column, data=data, y=target,
                        ax=ax, errorbar=None, color="red")
            customize_bar(axes=ax, position="v", 
                          values_font=values_font, pct=True, round_to=2)
        else:
            sns.countplot(x=column, data=data, ax=ax, color="red")
            customize_bar(axes=ax, position="v", values_font=values_font)

        ax.set_xlabel("")
        ax.set_title(column.capitalize(), fontsize=12)

        if i % col_number != 0:
            ax.set_ylabel("")

        ax.set_xticks(data[column].unique())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate)
    return fig, plt

            
def plot_num_dist(columns: List[str], 
                  data: pd.DataFrame, 
                  figsize: Tuple[int, int], 
                  title: str, 
                  plot_type: str = "hist", 
                  hue:Optional[str] = None,
                 bins:Union[str, int] = "auto") -> None:
    """
    Plots distributions for numerical columns using either histograms or box plots.
    
    Parameters:
    columns : List[str]
        A list of numerical column names to be plotted from the DataFrame.
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    figsize : Tuple[int, int]
        The size of the figure (width, height) in inches.
    title : str
        The title for the entire figure, displayed at the top of the plot.
    plot_type : str, optional, default="hist"
        The type of plot to display. Can be "hist" for histograms or "box" for box plots.
    hue : Optional[str], optional, default=None
        A categorical column to add a color dimension to the plots. 
        This will group the data by the unique values in this column and 
        use different colors for each group.

    Returns:
        None
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, y=0.93)
    for i, column in enumerate(columns):
        ax = fig.add_subplot(math.ceil(len(columns) / 2), 2, i+1)
        if plot_type == "hist":
            sns.histplot(x=column, data=data, bins=bins,
                         ax=ax, color="blue", hue=hue)
        elif plot_type == "box":
            sns.boxplot(x=hue, y=column,
                        data=data, ax=ax, color="blue")  
        elif plot_type == "bar":
            sns.barplot(x=hue, y=column,
                        data=data, ax=ax, color="red", errorbar=None)  
            customize_bar(position="v", axes=ax, round_to=1)
    sns.despine();

def plot_counts(ycolumn: str, xcolumn: str, 
                data: pd.DataFrame, fmt: str, 
                title: str = "", ax: Optional[plt.Axes] = None) -> None:
    """
    Plots a heatmap representing the percentage distribution of the values 
    in a cross-tabulation between two categorical columns in the DataFrame.

    Parameters:
    ycolumn : str
        The name of the column to be plotted on the y-axis.
    xcolumn : str
        The name of the column to be plotted on the x-axis.
    data : pd.DataFrame
        The DataFrame containing the data.
    fmt : str
        The format string for annotations in the heatmap (e.g., ".2f" for 
        two decimal places).
    title : str, optional, default=""
        The title of the plot.
    ax : Optional[plt.Axes], optional, default=None
        A matplotlib Axes object where the heatmap will be plotted. If not provided, 
        the current active Axes will be used.

    Returns:
        sns.heatmap
    """
    count = pd.crosstab(data[ycolumn], data[xcolumn])
    count_pct = count.apply(lambda x: x / x.sum(), axis=1).fillna(0)
    plt.title(title, fontsize=16)
    if not ax:
        ax = plt.gca()
    heatmap = sns.heatmap(count_pct, cmap="coolwarm", 
                          annot=True, fmt=fmt, 
                          cbar=None, ax=ax)
    heatmap.set_title(title)
    return heatmap

def feature_elimination(X_train, y_train, X_val, y_val,
                        model_params, num_features_to_drop=10, max_iter=10):
    """
    Performs recursive feature elimination based on LightGBM feature importance.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature data.
    y_train : pd.Series or np.ndarray
        Training target data.
    X_val : pd.DataFrame
        Validation feature data.
    y_val : pd.Series or np.ndarray
        Validation target data.
    model_params : dict
        Dictionary of parameters for the LightGBM model.
    num_features_to_drop : int, optional (default=10)
        Number of least important features to drop in each iteration.
    max_iter : int, optional (default=10)
        Maximum number of iterations to run the feature elimination process.

    Returns:
    --------
    features_count_res : pd.DataFrame
        DataFrame containing AUC scores and dropped features for each iteration.
    X_reduced_train : pd.DataFrame
        Reduced training dataset with important features only.
    X_reduced_val : pd.DataFrame
        Reduced validation dataset with important features only.
    """
    features_count_res = pd.DataFrame(columns=["Score", "Dropped_Features"])

    X_reduced_train = X_train.copy()
    X_reduced_val = X_val.copy()

    for i in range(max_iter):
        print(f"Training with {len(X_reduced_train.columns)} features")

        # Create LightGBM datasets
        X_train_lgbm_dataset = lgb.Dataset(X_reduced_train, label=y_train, categorical_feature="auto")
        X_val_lgbm_dataset = lgb.Dataset(X_reduced_val, label=y_val, reference=X_train_lgbm_dataset)

        # Train the LightGBM model
        model = lgb.train(
            model_params,
            X_train_lgbm_dataset,
            valid_sets=[X_train_lgbm_dataset, X_val_lgbm_dataset],
            valid_names=['train', 'val'],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )

        feature_importances = pd.DataFrame({
            'feature': X_reduced_train.columns,
            'importance': model.feature_importance()
        }).sort_values(by='importance')

        least_important_features = feature_importances.head(num_features_to_drop)['feature']

        num_features = len(X_reduced_train.columns)
        features_count_res.loc[num_features, "Score"] = model.best_score["val"]["auc"]
        features_count_res.loc[num_features, "Dropped_Features"] = least_important_features.values

        X_reduced_train = X_reduced_train.drop(columns=least_important_features)
        X_reduced_val = X_reduced_val.drop(columns=least_important_features)

    return features_count_res


def plot_missing_values_by_row(dataset: pd.DataFrame, figsize=(6, 4)):
    """
    This function plots the number of rows with missing values by their count,
    and customizes the plot using helper_functions.

    Parameters:
    dataset (pd.DataFrame): The DataFrame for which missing values per row are to be analyzed.
    """
    missing_count_byrow = (dataset.isnull().sum(axis=1).value_counts().sort_index())
    plt.figure(figsize=figsize)
    missing_count_byrow.plot(kind="barh", color="red")

    ax = plt.gca()
    ax.set_xlabel("Number of rows")
    plt.title("Number of rows for each count of missing values", fontsize=16)

    customize_bar(axes=ax, position="h", values_font=10)


def missing_values_pattern(dataset:pd.DataFrame, columns: List[str]) -> pd.Series:
    """
    This function identifies missing values in some group of columns,
    and returns the count of rows with different missing value patterns.

    Parameters:
    dataset (pd.DataFrame): The DataFrame in which to check for missing values.
    columns (str): The columns, for which the missing data will be checked.

    Returns:
    pd.Series: A series showing the count of rows with each missing value pattern.
    """

    missing_pattern = dataset[columns].isnull()

    return missing_pattern.sum(axis=1).value_counts()


def print_unique_values_for_cat_columns(df: pd.DataFrame):
    cat_cols = df.select_dtypes("object").columns

    for column in cat_cols:
        unique_values = df[column].unique()
        print("---------------------------------------------")
        print(column)
        print(unique_values)


def remove_rare_categories(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    This function standardizes rare categories in specific columns of the dataset by
    grouping them under common categories or renaming them as 'Other'.

    Parameters:
    dataset (pd.DataFrame): The dataset to be processed.

    Returns:
    pd.DataFrame: The dataset with rare categories replaced.
    """

    rare_income_types = ["Businessman", "Student",
                         "Maternity leave", "Unemployed", "Pensioner"]
    dataset.loc[dataset["NAME_INCOME_TYPE"].isin(rare_income_types), "NAME_INCOME_TYPE"] = "Other"

    rare_occupation_types = ["Private service staff", "IT staff",
                             "HR staff", "Realty agents", "Secretaries",
                             "Waiters/barmen staff", "Low-skill Laborers"]
    dataset.loc[dataset["OCCUPATION_TYPE"].isin(rare_occupation_types), "OCCUPATION_TYPE"] = "Other"

    organization_group = ["Industry", "Business Entity", "Trade", "Transport"]
    for group in organization_group:
        dataset.loc[
            dataset["ORGANIZATION_TYPE"].str.startswith(group), "ORGANIZATION_TYPE"] = group

    rare_organizations = ["Religion", "Cleaning", "Legal Services",
                          "Mobile", "Culture", "Realtor", "Advertising",
                          "Emergency", "Telecom", "Insurance", "Electricity",
                          "Hotel", "Military", "Bank", "Agriculture", "Police",
                          "Postal", "Security Ministries", "Restaurant", "University",
                          "Services", "Restaurant", "Housing"]
    dataset.loc[dataset["ORGANIZATION_TYPE"].isin(rare_organizations), "ORGANIZATION_TYPE"] = "Other"

    rare_accompanied = ["Group of people", "Other_A", "UNKNOWN", "Other_B"]
    dataset.loc[dataset["NAME_TYPE_SUITE"].isin(rare_accompanied), "NAME_TYPE_SUITE"] = "Other"

    rare_housing = ["Co-op apartment", "Office apartment"]
    dataset.loc[dataset["NAME_HOUSING_TYPE"].isin(rare_housing), "NAME_HOUSING_TYPE"] = "Other"

    dataset.loc[dataset["CODE_GENDER"] == "UNKNOWN", "CODE_GENDER"] = "F"

    dataset.loc[dataset["NAME_FAMILY_STATUS"] == "UNKNOWN", "NAME_FAMILY_STATUS"] = "Married"

    return dataset
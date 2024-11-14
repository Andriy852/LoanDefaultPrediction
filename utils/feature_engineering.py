import pandas as pd
from typing import List, Tuple
import numpy as np

def create_application_feature(df: pd.DataFrame,
                               ext_source_cols: List[str]) -> pd.DataFrame:
    """
    This function creates new features from the original application table.
    In addition it claps outliers for some columns.
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset on which the features are created.
    ext_source_cols : List[str]
        The list of columns related to external source information for averaging/summing.
    Returns:
    --------
    df : pd.DataFrame
        The dataset with new features added.
    """

    df["DEBT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["INCOME_PER_ANNUITY_UNIT"] = df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
    df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    df["LIVES_IN_POPULATED_AREA"] = (df["REGION_POPULATION_RELATIVE"] > 0.04).astype(int)
    df["FRIEND_DEFAULTED"] = ((df["DEF_60_CNT_SOCIAL_CIRCLE"] > 0) |
                              (df["DEF_30_CNT_SOCIAL_CIRCLE"] > 0)).astype(int)
    df["FRIEND_HAD_DPD"] = ((df["OBS_30_CNT_SOCIAL_CIRCLE"] > 0) |
                            (df["OBS_60_CNT_SOCIAL_CIRCLE"] > 0)).astype(int)
    df["MISSING_BUREAU"] = df["AMT_REQ_CREDIT_BUREAU_WEEK"].isna().astype(int)

    df["HAD_ENQUIRIES"] = (
            (df["AMT_REQ_CREDIT_BUREAU_HOUR"] > 0) |
            (df["AMT_REQ_CREDIT_BUREAU_DAY"] > 0) |
            (df["AMT_REQ_CREDIT_BUREAU_WEEK"] > 0) |
            (df["AMT_REQ_CREDIT_BUREAU_MON"] > 0) |
            (df["AMT_REQ_CREDIT_BUREAU_QRT"] > 0) |
            (df["AMT_REQ_CREDIT_BUREAU_YEAR"] > 0)
    ).astype(int)
    df["AVG_EXT_SOURCE"] = df[ext_source_cols].mean(axis=1)
    df["SUM_EXT_SOURCE"] = df[ext_source_cols].sum(axis=1)

    df.loc[:, "CNT_CHILDREN"] = (df["CNT_CHILDREN"]
                                 .apply(lambda x: 3 if x >= 3 else x))
    df.loc[:, "CNT_FAM_MEMBERS"] = (df["CNT_FAM_MEMBERS"]
                                    .apply(lambda x: 4 if x >= 4 else x))

    return df

def merge_application_features(
        main_data: pd.DataFrame,
        prev_application_new_features: pd.DataFrame,
        application_bureau_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merges new application features and bureau features into the main dataset.

    Parameters:
    - main_data: The main dataset (X or test_data) to which features will be added.
    - prev_application_new_features: DataFrame containing new application features to merge.
    - application_bureau_features: DataFrame containing bureau features to merge.

    Returns:
    - DataFrame: The main dataset with the new features added and null values replaced with 0.
    """

    main_data = main_data.merge(prev_application_new_features, on="SK_ID_CURR", how="left")
    main_data.loc[main_data["APPLICATION_COUNT_PREV"].isnull(), "APPLICATION_COUNT_PREV"] = 0

    main_data = main_data.merge(application_bureau_features, on="SK_ID_CURR", how="left")
    main_data.loc[main_data["APPLICATION_COUNT_BUREAU"].isnull(), "APPLICATION_COUNT_BUREAU"] = 0

    return main_data

def drop_correlated_features(
        main_data: pd.DataFrame,
        test_data: pd.DataFrame,
        columns_to_drop: List[str],
        categorical_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the main and test datasets by dropping strongly correlated columns and handling
    "Unemployed" values in categorical columns.

    Parameters:
    - main_data: The main dataset (X) to preprocess.
    - test_data: The test dataset to preprocess.
    - columns_to_drop: List of column names to drop from both datasets.
    - categorical_columns: List of categorical columns to handle "Unemployed" values.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Updated main_data and test_data.
    """

    main_data.drop(columns=columns_to_drop, inplace=True)
    test_data.drop(columns=columns_to_drop, inplace=True)

    for col in categorical_columns:
        main_data.loc[main_data[col] == "Unemployed", col] = np.nan
        test_data.loc[test_data[col] == "Unemployed", col] = np.nan

    return main_data, test_data


def create_features_from_previous_data(pos_cash: pd.DataFrame, installments: pd.DataFrame,
                                       credit_card: pd.DataFrame,
                                       previous_application: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from POS_CASH_balance, installments_payments, and credit_card_balance tables
    and merges them into the previous_application table.

    Parameters:
    ----------
    pos_cash : pd.DataFrame
        DataFrame containing monthly records for POS and cash loan balances,
        with each record associated with a previous application.

    installments : pd.DataFrame
        DataFrame containing monthly records for installment payments for each previous application.

    credit_card : pd.DataFrame
        DataFrame containing monthly records for revolving credit card balances and payments.

    previous_application : pd.DataFrame
        DataFrame of previous applications containing general information on each application.

    Returns:
    -------
    pd.DataFrame
        The previous_application DataFrame with additional features created from the three input tables.

    Additional Information:
    -----------------------
    - The function groups data by `SK_ID_PREV` for each input table to create summary features.
    - The function then merges these new features into the `previous_application` DataFrame.
    - Any column name conflicts in the merges are handled by filling missing values
      from one column with values from the other, then dropping duplicates.

    Example Features Created:
    -------------------------
    - Loan duration in months
    - Whether the loan was completed or not
    - Average and last monthly snapshots of loan balances
    - Whether there were any late payments or underpaid installments
    """
    finished_status_pos_cash = ["Canceled", "Returned to the store", "Completed"]
    finished_status_credit_card = ["Refused", "Completed"]

    agg_pos_cash = pd.concat([
        pos_cash.groupby("SK_ID_PREV").apply(
            lambda x: (
                x["MONTHS_BALANCE"].max() - x["MONTHS_BALANCE"].min() + 1
                if (x["NAME_CONTRACT_STATUS"] == "Completed").any()
                else (
                        x["MONTHS_BALANCE"].max() - x["MONTHS_BALANCE"].min() + 1
                        + (
                            x.loc[x["MONTHS_BALANCE"] == x["MONTHS_BALANCE"].max(),
                            "CNT_INSTALMENT_FUTURE"].iloc[0]
                            if not x["CNT_INSTALMENT_FUTURE"].isnull().all()
                            else 0
                        )
                )
            ), include_groups=False),
        pos_cash.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["NAME_CONTRACT_STATUS"].isin(finished_status_pos_cash)).any()
            else "NO", include_groups=False),
        pos_cash.groupby("SK_ID_PREV")["MONTHS_BALANCE"].max(),
        pos_cash.groupby("SK_ID_PREV")["MONTHS_BALANCE"].mean(),
        pos_cash.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if len(x["CNT_INSTALMENT"].unique()) > 1 else "NO",
            include_groups=False),
        pos_cash.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["SK_DPD"] != 0).any() else "NO",
            include_groups=False)
    ], axis=1)

    agg_pos_cash.columns = ["LOAN_LENGTH", "IS_COMPLETED", "LAST_SNAPSHOT", "AVERAGE_SNAPSHOT",
                            "INSTALLMENT_CHANGED", "HAD_DPD"]
    agg_pos_cash.reset_index(inplace=True)

    agg_installments = pd.concat([
        installments.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if len(x["NUM_INSTALMENT_VERSION"].unique()) != 1
            else "NO", include_groups=False),
        installments.groupby("SK_ID_PREV")["AMT_INSTALMENT"].mean(),
        installments.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["DAYS_ENTRY_PAYMENT"] > x["DAYS_INSTALMENT"]).any()
            else "NO", include_groups=False),
        installments.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["AMT_PAYMENT"] < x["AMT_INSTALMENT"]).any()
            else "NO", include_groups=False),
        installments.groupby("SK_ID_PREV").size()
    ], axis=1)

    agg_installments.columns = ["VERSION_CHANGED", "AVERAGE_INSTALLMENT",
                                "HAD_DPD", "UNDERPAID", "INSTALLMENT_COUNT"]
    agg_installments.reset_index(inplace=True)

    credit_card = credit_card[credit_card["AMT_DRAWINGS_CURRENT"] >= 0]
    agg_credit_card = pd.concat([
        credit_card.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["NAME_CONTRACT_STATUS"].isin(finished_status_credit_card)).any()
            else "NO", include_groups=False),
        credit_card.groupby("SK_ID_PREV")["MONTHS_BALANCE"].max(),
        credit_card.groupby("SK_ID_PREV")["MONTHS_BALANCE"].mean(),
        credit_card.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["SK_DPD"] != 0).any() else "NO", include_groups=False),
        credit_card.groupby("SK_ID_PREV")["AMT_DRAWINGS_CURRENT"].mean(),
        credit_card.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["AMT_DRAWINGS_CURRENT"] > x["AMT_CREDIT_LIMIT_ACTUAL"]).any()
            else "NO", include_groups=False),
        credit_card.groupby("SK_ID_PREV").apply(
            lambda x: "YES" if (x["AMT_INST_MIN_REGULARITY"] > x["AMT_PAYMENT_CURRENT"]).any()
            else "NO", include_groups=False)
    ], axis=1)

    agg_credit_card.columns = ["IS_COMPLETED", "LAST_SNAPSHOT", "AVERAGE_SNAPSHOT",
                               "HAD_DPD", "AVG_DRAWINGS", "EXCEEDED_LIMIT", "UNDERPAID"]
    agg_credit_card.reset_index(inplace=True)

    previous_application = previous_application.merge(agg_installments, on="SK_ID_PREV", how="left")
    previous_application = previous_application.merge(agg_pos_cash, on="SK_ID_PREV", how="left")
    previous_application = previous_application.merge(agg_credit_card, on="SK_ID_PREV", how="left")

    previous_application['UNDERPAID'] = (previous_application['UNDERPAID_x']
                                         .fillna(previous_application['UNDERPAID_y']))
    previous_application.drop(columns=['UNDERPAID_x', 'UNDERPAID_y'], inplace=True)

    previous_application['IS_COMPLETED'] = (previous_application['IS_COMPLETED_x']
                                            .fillna(previous_application['IS_COMPLETED_y']))
    previous_application.drop(columns=['IS_COMPLETED_x', 'IS_COMPLETED_y'], inplace=True)

    previous_application['LAST_SNAPSHOT'] = (previous_application['LAST_SNAPSHOT_x']
                                             .fillna(previous_application['LAST_SNAPSHOT_y']))
    previous_application.drop(columns=['LAST_SNAPSHOT_x', 'LAST_SNAPSHOT_y'], inplace=True)

    previous_application['AVERAGE_SNAPSHOT'] = (previous_application['AVERAGE_SNAPSHOT_x']
                                                .fillna(previous_application['AVERAGE_SNAPSHOT_y']))
    previous_application.drop(columns=['AVERAGE_SNAPSHOT_x', 'AVERAGE_SNAPSHOT_y'], inplace=True)

    previous_application['HAD_DPD'] = (previous_application['HAD_DPD_x']
                                       .fillna(previous_application['HAD_DPD_y'])
                                       .fillna(previous_application['HAD_DPD']))
    previous_application.drop(columns=['HAD_DPD_x', 'HAD_DPD_y'], inplace=True)

    return previous_application


def fill_unapproved_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the `previous_application` DataFrame for applications
    that were not approved. Specifically, this function adjusts `IS_COMPLETED`,
    `INSTALLMENT_COUNT`, `CNT_PAYMENT`, and `LOAN_LENGTH` based on the status
    of the application.

    Parameters:
    ----------
    previous_application : pd.DataFrame
        DataFrame containing details of previous applications, including loan status,
        installment count, contract status, and loan length.

    Returns:
    -------
    pd.DataFrame
        The modified `previous_application` DataFrame with filled values for
        unapproved applications.
    """

    unappr_mask = previous_application["NAME_CONTRACT_STATUS"] != "Approved"

    previous_application.loc[
        (unappr_mask) & (previous_application["IS_COMPLETED"] == "UNKNOWN"),
        "IS_COMPLETED"] = "YES"

    previous_application.loc[
        (unappr_mask) & (previous_application["INSTALLMENT_COUNT"].isnull()),
        "INSTALLMENT_COUNT"] = 0

    previous_application.loc[
        (unappr_mask) & (previous_application["CNT_PAYMENT"].isnull()),
        "CNT_PAYMENT"] = 0

    previous_application.loc[
        (unappr_mask) & (previous_application["LOAN_LENGTH"].isnull()),
        "LOAN_LENGTH"] = 0

    length_miss = previous_application.loc[previous_application["LOAN_LENGTH"].isnull()]
    months_diff = np.ceil((length_miss["DAYS_TERMINATION"] - length_miss["DAYS_DECISION"]) / 30.44)
    previous_application["LOAN_LENGTH"] = previous_application["LOAN_LENGTH"].fillna(months_diff)

    return previous_application

def aggregate_client_level(
    previous_application: pd.DataFrame,
    pos_cash: pd.DataFrame,
    credit_card: pd.DataFrame,
    installments: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates features from the `previous_application`, `pos_cash`,
    `credit_card`, and `installments` DataFrames at the client level
    based on `SK_ID_CURR`.

    Parameters:
    ----------
    previous_application : pd.DataFrame
        DataFrame containing information on clients' previous applications.
    pos_cash : pd.DataFrame
        DataFrame with POS cash balance history.
    credit_card : pd.DataFrame
        DataFrame with credit card balance history.
    installments : pd.DataFrame
        DataFrame with installment payment history.

    Returns:
    -------
    pd.DataFrame
        Aggregated DataFrame at the client level (`SK_ID_CURR`), containing
        client-level features.
    """

    agg_credit_funcs = {
        "AVERAGE_DRAWINGS_PREV": ("AMT_DRAWINGS_CURRENT", "mean"),
        "AVG_MONTHS_BALANCE": ("MONTHS_BALANCE", "mean")
    }
    agg_credit_card_client = credit_card.groupby("SK_ID_CURR").agg(**agg_credit_funcs)

    agg_pos_cash_funcs = {
        "DPD_COUNT": ("SK_DPD", "sum"),
        "AVG_MONTHS_BALANCE": ("MONTHS_BALANCE", "mean")
    }
    agg_pos_cash_client = pos_cash.groupby("SK_ID_CURR").agg(**agg_pos_cash_funcs)

    agg_installment_funcs = {
        "AVERAGE_INSTALLMENT_PREV": ("AMT_INSTALMENT", "mean")
    }
    agg_installments_client = installments.groupby("SK_ID_CURR").agg(**agg_installment_funcs)

    agg_funcs = {
        "APPLICATION_COUNT_PREV": ("SK_ID_CURR", "size"),
        "HAD_NON_APPROVED_PREV": ("NAME_CONTRACT_STATUS",
                                  lambda x: "YES" if (x != "Approved").any() else "NO"),
        "HAD_CONSUMER_LOAN_PREV": ("NAME_CONTRACT_TYPE",
                                   lambda x: "YES" if (x == 'Consumer loans').any() else "NO"),
        "HAD_REVOLVING_LOAN_PREV": ("NAME_CONTRACT_TYPE",
                                    lambda x: "YES" if (x == 'Revolving loans').any() else "NO"),
        "HAD_CASH_LOAN_PREV": ("NAME_CONTRACT_TYPE",
                               lambda x: "YES" if (x == 'Cash loans').any() else "NO"),
        "HAD_X_SELL_PREV": ("NAME_PRODUCT_TYPE",
                            lambda x: "YES" if (x == 'x-sell').any() else "NO"),
        "HAS_ONGOING_LOAN_PREV": ("IS_COMPLETED",
                                  lambda x: "YES" if (x != "NO").any() else "NO"),
        "HAD_DPD_PREV": ("HAD_DPD",
                         lambda x: "YES" if (x == "YES").any() else "NO"),
        "VERSION_CHANGED_PREV": ("VERSION_CHANGED",
                                 lambda x: "YES" if (x == "YES").any() else "NO"),
        "LIMIT_EXCEEDED_PREV": ("EXCEEDED_LIMIT",
                                lambda x: "YES" if (x == "YES").any() else "NO"),
        "UNDERPAID_PREV": ("UNDERPAID",
                           lambda x: "YES" if (x == "YES").any() else "NO"),
        "INSTALLMENTS_OVER_CNT_PAYMENT_PREV": (
            "INSTALLMENT_COUNT",
            lambda x: 'YES' if (x > previous_application.loc[x.index, "CNT_PAYMENT"]).any()
            else "NO"
        ),
        "AVERAGE_YIELD_GROUP_PREV": ("NAME_YIELD_GROUP", "mean"),
        "AVERAGE_LOAN_LENGTH_PREV": ("LOAN_LENGTH", "mean"),
        "AVERAGE_CNT_PAYMENT_PREV": ("CNT_PAYMENT", "mean"),
        "LAST_SNAPSHOT_MAX_PREV": ("LAST_SNAPSHOT", "max"),
        "LAST_SNAPSHOT_MEAN_PREV": ("LAST_SNAPSHOT", "mean"),
        "DAYS_DECISION_MEAN_PREV": ("DAYS_DECISION", "mean"),
        "DAYS_DECISION_MAX_PREV": ("DAYS_DECISION", "max"),
        "INSTALLMENT_COUNT_MEAN_PREV": ("INSTALLMENT_COUNT", "mean"),
        "AVERAGE_AMT_CREDIT_PREV": ("AMT_CREDIT", "mean"),
        "AVERAGE_RATE_DOWN_PAYMENT_PREV": ("RATE_DOWN_PAYMENT", "mean")
    }
    agg_previous_application = previous_application.groupby("SK_ID_CURR").agg(**agg_funcs)
    agg_previous_application.reset_index(inplace=True)

    agg_prev = agg_previous_application.merge(agg_pos_cash_client, on="SK_ID_CURR", how="left")
    agg_prev = agg_prev.merge(agg_credit_card_client, on="SK_ID_CURR", how="left")
    agg_prev = agg_prev.merge(agg_installments_client, on="SK_ID_CURR", how="left")

    agg_prev['AVG_MONTHS_BALANCE'] = agg_prev['AVG_MONTHS_BALANCE_x'].fillna(
        agg_prev['AVG_MONTHS_BALANCE_y'])
    agg_prev.drop(columns=['AVG_MONTHS_BALANCE_x', 'AVG_MONTHS_BALANCE_y'], inplace=True)

    return agg_prev


def aggregate_bureau_balance(bureau_balance: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates features from the 'bureau_balance' table and merges them with the 'bureau' table.

    Parameters:
    - bureau_balance (pd.DataFrame): The dataframe containing bureau balance data.
    - bureau (pd.DataFrame): The dataframe containing bureau data to merge
    the aggregated features into.

    Returns:
    - pd.DataFrame: The merged dataframe containing aggregated bureau balance features.
    """

    curr_status = bureau_balance.groupby("SK_ID_BUREAU").apply(
        lambda x: x.loc[x["MONTHS_BALANCE"].idxmax(), "STATUS"]
    )
    curr_status = pd.DataFrame(curr_status).rename(columns={0: "CURRENT_STATUS"})

    agg_funcs = {
        "LAST_SNAPSHOT": ("MONTHS_BALANCE", "max"),
        "HAD_1_DPD": ("STATUS", lambda x: "YES" if (x == "1").any() else "NO"),
        "HAD_2_DPD": ("STATUS", lambda x: "YES" if (x == "2").any() else "NO"),
        "HAD_3_DPD": ("STATUS", lambda x: "YES" if (x == "3").any() else "NO"),
        "HAD_4_DPD": ("STATUS", lambda x: "YES" if (x == "4").any() else "NO"),
        "HAD_5_DPD": ("STATUS", lambda x: "YES" if (x == "5").any() else "NO")
    }

    agg_bureau_balance = bureau_balance.groupby("SK_ID_BUREAU").agg(**agg_funcs)

    agg_bureau_balance = agg_bureau_balance.join(curr_status, on="SK_ID_BUREAU")

    bureau = bureau.merge(agg_bureau_balance, on="SK_ID_BUREAU", how="left")

    return bureau

def aggregate_bureau_data(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates bureau data at the SK_ID_CURR level.

    Args:
        bureau (pd.DataFrame): The bureau data containing customer credit information.

    Returns:
        pd.DataFrame: Aggregated bureau data at the SK_ID_CURR level.
    """
    agg_funcs = {
        "APPLICATION_COUNT_BUREAU": ("SK_ID_CURR", "size"),
        "HAD_CONSUMER_LOAN_BUREAU": ("CREDIT_TYPE",
                                     lambda x: "YES" if (x == 'Consumer credit').any() else "NO"),
        "HAD_REVOLVING_LOAN_BUREAU": ("CREDIT_TYPE",
                                      lambda x: "YES" if (x == 'Credit card').any() else "NO"),
        "HAS_ONGOING_LOAN_BUREAU": ("CREDIT_ACTIVE",
                                    lambda x: "YES" if (x == "Active").any() else "NO"),
        "HAD_PROLONGATION_BUREAU": ("CNT_CREDIT_PROLONG",
                                    lambda x: "YES" if (x != 0).any() else "NO"),

        "HAD_DPD_BUREAU": ("CREDIT_DAY_OVERDUE",
                           lambda x: "YES" if (x != 0).any() else "NO"),
        "DPD_SUM_BUREAU": ("CREDIT_DAY_OVERDUE", "sum"),
        "HAD_1DPD_BUREAU": ("HAD_1_DPD",
                            lambda x: "YES" if (x == "YES").any() else "NO"),
        "HAD_2DPD_BUREAU": ("HAD_2_DPD",
                            lambda x: "YES" if (x == "YES").any() else "NO"),
        "HAD_3DPD_BUREAU": ("HAD_3_DPD",
                            lambda x: "YES" if (x == "YES").any() else "NO"),
        "HAD_4DPD_BUREAU": ("HAD_4_DPD",
                            lambda x: "YES" if (x == "YES").any() else "NO"),
        "HAD_5DPD_BUREAU": ("HAD_5_DPD",
                            lambda x: "YES" if (x == "YES").any() else "NO"),

        "HAD_DEBT_BUREAU": ("AMT_CREDIT_SUM_DEBT",
                            lambda x: "YES" if (x != 0).any() else "NO"),
        "SUM_AMT_DEBT_BUREAU": ("AMT_CREDIT_SUM_DEBT", "sum"),

        "SUM_AMT_OVERDUE_BUREAU": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        "HAD_AMT_OVERDUE_BUREAU": ("AMT_CREDIT_SUM_OVERDUE",
                                   lambda x: "YES" if (x != 0).any() else "NO"),

        "AVG_AMT_CREDIT_SUM_BUREAU": ("AMT_CREDIT_SUM", "mean"),
        "MAX_AMT_CREDIT_SUM_BUREAU": ("AMT_CREDIT_SUM", "max"),
        "MAX_AMT_CREDIT_MAX_OVERDUE_BUREAU": ("AMT_CREDIT_MAX_OVERDUE", "max"),
        "LAST_SNAPSHOT_MAX_BUREAU": ("LAST_SNAPSHOT", "max"),
        "LAST_SNAPSHOT_MEAN_BUREAU": ("LAST_SNAPSHOT", "mean"),
        "AVG_DAYS_CREDIT_BUREAU": ("DAYS_CREDIT", "mean"),
        "MAX_DAYS_CREDIT_BUREAU": ("DAYS_CREDIT", "max"),
        "AVG_DAYS_ENDDATE_FACT_BUREAU": ("DAYS_ENDDATE_FACT", "mean"),
        "MAX_DAYS_ENDDATE_FACT_BUREAU": ("DAYS_ENDDATE_FACT", "max"),
        "AVG_LOAN_LENGTH_BUREAU": ("LOAN_LENGTH", "mean"),
    }

    agg_bureau = bureau.groupby("SK_ID_CURR").agg(**agg_funcs)
    return agg_bureau


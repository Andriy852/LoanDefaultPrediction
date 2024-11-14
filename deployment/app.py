from fastapi import FastAPI
from loan_application import Client_Data, PredictionOut
import joblib
import pandas as pd
from utils.preprocessing import prepare_categorical_datasets

app = FastAPI()

log_reg_model = joblib.load('log_reg_pipeline.pkl')
lgb_model = joblib.load('lgbm_model.pkl')

zero_importance_features = ['HAD_2DPD_BUREAU', 'HAD_REVOLVING_LOAN_BUREAU',
                            'LIVES_IN_POPULATED_AREA', 'HAD_3DPD_BUREAU', 'HAD_4DPD_BUREAU',
                            'HAD_5DPD_BUREAU', 'HAD_DEBT_BUREAU', 'FLAG_DOCUMENT_20',
                            'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_21', 'HAD_CONSUMER_LOAN_BUREAU',
                            'REG_REGION_NOT_WORK_REGION', 'FLAG_EMAIL',
                            'REG_REGION_NOT_LIVE_REGION', 'HAD_ENQUIRIES', 'FLAG_DOCUMENT_19',
                            'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_17', 'HAD_CONSUMER_LOAN_PREV',
                            'FLAG_DOCUMENT_16', 'HAD_CASH_LOAN_PREV', 'HAS_ONGOING_LOAN_PREV',
                            'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_14',
                            'INSTALLMENTS_OVER_CNT_PAYMENT_PREV', 'FLAG_DOCUMENT_13',
                            'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_6',
                            'FLAG_DOCUMENT_5', 'REG_CITY_NOT_WORK_CITY',
                            'LIVE_REGION_NOT_WORK_REGION', 'MISSING_BUREAU',
                            'LIVE_CITY_NOT_WORK_CITY']
drop_columns = zero_importance_features

@app.post("/lgbm_predict", response_model=PredictionOut)
def predict(client_data: Client_Data):
    dataframe = pd.DataFrame([client_data.dict()])
    dataframe = prepare_categorical_datasets(dataframe,
                                             drop_column=drop_columns)
    preds = lgb_model.predict(dataframe)
    result = {"DefaultProbability": preds}
    return result

@app.post("/log_reg_predict", response_model=PredictionOut)
def predict(client_data: Client_Data):
    dataframe = pd.DataFrame([client_data.dict()])
    preds = log_reg_model.predict_proba(dataframe)[0, 1]
    result = {"DefaultProbability": preds}
    return result


from pydantic import BaseModel
from typing import Literal

class Client_Data(BaseModel):
    NAME_CONTRACT_TYPE: Literal['Cash loans', 'Revolving loans']
    CODE_GENDER: Literal['M', 'F']
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    NAME_TYPE_SUITE: Literal['Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other']
    NAME_INCOME_TYPE: Literal['Working', 'State servant', 'Commercial associate', 'Other']
    NAME_EDUCATION_TYPE: Literal[
        'Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    NAME_FAMILY_STATUS: Literal['Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated']
    NAME_HOUSING_TYPE: Literal['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Other']
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: Literal[
        'Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff', 'UNKNOWN', 'Other', 'Medicine staff', 'Security staff', 'High skill tech staff']
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: Literal['WEDNESDAY', 'MONDAY', 'THURSDAY', 'SUNDAY', 'SATURDAY', 'FRIDAY', 'TUESDAY']
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: Literal['Business Entity', 'School', 'Government', 'Other', 'Medicine', 'Self-employed', 'Transport', 'Construction', 'Kindergarten', 'Trade', 'Industry', 'Security']
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    INDICATOR_SUM: int
    DOCUMENT_SUM: int
    DEBT_TO_INCOME: float
    CREDIT_TERM: float
    INCOME_PER_ANNUITY_UNIT: float
    LIVES_IN_POPULATED_AREA: int
    FRIEND_DEFAULTED: int
    FRIEND_HAD_DPD: int
    MISSING_BUREAU: int
    HAD_ENQUIRIES: int
    AVG_EXT_SOURCE: float
    SUM_EXT_SOURCE: float
    APPLICATION_COUNT_PREV: float
    HAD_NON_APPROVED_PREV: Literal['NO', 'YES']
    HAD_CONSUMER_LOAN_PREV: Literal['YES', 'NO']
    HAD_REVOLVING_LOAN_PREV: Literal['NO', 'YES']
    HAD_CASH_LOAN_PREV: Literal['NO', 'YES']
    HAD_X_SELL_PREV: Literal['NO', 'YES']
    HAS_ONGOING_LOAN_PREV: Literal['NO', 'YES']
    HAD_DPD_PREV: Literal['NO', 'YES']
    VERSION_CHANGED_PREV: Literal['YES', 'NO']
    LIMIT_EXCEEDED_PREV: Literal['NO', 'YES']
    UNDERPAID_PREV: Literal['NO', 'YES']
    INSTALLMENTS_OVER_CNT_PAYMENT_PREV: Literal['NO', 'YES']
    AVERAGE_YIELD_GROUP_PREV: float
    AVERAGE_LOAN_LENGTH_PREV: float
    AVERAGE_CNT_PAYMENT_PREV: float
    LAST_SNAPSHOT_MAX_PREV: float
    LAST_SNAPSHOT_MEAN_PREV: float
    DAYS_DECISION_MEAN_PREV: float
    DAYS_DECISION_MAX_PREV: float
    INSTALLMENT_COUNT_MEAN_PREV: float
    AVERAGE_AMT_CREDIT_PREV: float
    AVERAGE_RATE_DOWN_PAYMENT_PREV: float
    DPD_COUNT: float
    AVERAGE_DRAWINGS_PREV: float
    AVERAGE_INSTALLMENT_PREV: float
    AVG_MONTHS_BALANCE: float
    APPLICATION_COUNT_BUREAU: float
    HAD_CONSUMER_LOAN_BUREAU: Literal['YES', 'NO']
    HAD_REVOLVING_LOAN_BUREAU: Literal['YES', 'NO']
    HAS_ONGOING_LOAN_BUREAU: Literal['YES', 'NO']
    HAD_PROLONGATION_BUREAU: Literal['NO', 'YES']
    HAD_DPD_BUREAU: Literal['NO', 'YES']
    HAD_1DPD_BUREAU: Literal['YES', 'NO']
    HAD_2DPD_BUREAU: Literal['NO', 'YES']
    HAD_3DPD_BUREAU: Literal['NO', 'YES']
    HAD_4DPD_BUREAU: Literal['NO', 'YES']
    HAD_5DPD_BUREAU: Literal['NO', 'YES']
    HAD_DEBT_BUREAU: Literal['YES', 'NO']
    SUM_AMT_DEBT_BUREAU: float
    AVG_AMT_CREDIT_SUM_BUREAU: float
    MAX_AMT_CREDIT_SUM_BUREAU: float
    MAX_AMT_CREDIT_MAX_OVERDUE_BUREAU: float
    LAST_SNAPSHOT_MAX_BUREAU: float
    LAST_SNAPSHOT_MEAN_BUREAU: float
    AVG_DAYS_CREDIT_BUREAU: float
    MAX_DAYS_CREDIT_BUREAU: float
    AVG_DAYS_ENDDATE_FACT_BUREAU: float
    MAX_DAYS_ENDDATE_FACT_BUREAU: float
    AVG_LOAN_LENGTH_BUREAU: float

class PredictionOut(BaseModel):
    DefaultProbability: float
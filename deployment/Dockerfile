FROM python:3.9.13-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY ["deployment/lgbm_model.pkl", "deployment/log_reg_pipeline.pkl", "deployment/app.py", "deployment/loan_application.py", "./"]
COPY utils/ /app/utils/

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
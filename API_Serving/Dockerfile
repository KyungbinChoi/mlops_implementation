FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install mlflow==1.30.0 pandas scikit-learn "fastapi[all]"

COPY schemas.py schemas.py
COPY app.py app.py
COPY rf_model/ rf_model/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload"]
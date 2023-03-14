from argparse import ArgumentParser
import pandas as pd
import psycopg2
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import os

# setting env arguements for mlflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. get data
db_connect = psycopg2.connect(host="localhost", database="mydatabase", user="myuser", password="mypassword")
df = pd.read_sql("SELECT * FROM housing_data", db_connect)
X = df.drop(["id", "medhouseval"], axis="columns")
y = df["medhouseval"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

# 2. model development and train
knn_pipeline = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsRegressor(n_neighbors=5))])
knn_pipeline.fit(X_train, y_train)
knn_train_pred = knn_pipeline.predict(X_train)
knn_valid_pred = knn_pipeline.predict(X_valid)

rf_pipeline = Pipeline([("rf", RandomForestRegressor(n_estimators=250,max_depth=20))])
rf_pipeline.fit(X_train, y_train)
rf_train_pred = rf_pipeline.predict(X_train)
rf_valid_pred = rf_pipeline.predict(X_valid)

train_rmse_knn = mean_squared_error(y_true=y_train, y_pred=knn_train_pred, squared=False)
train_rmse_rf = mean_squared_error(y_true=y_train, y_pred=rf_train_pred, squared=False)
print("Train RMSE using KNN:", train_rmse_knn)
print("Train RMSE using Randomforest:", train_rmse_rf)

valid_rmse_knn = mean_squared_error(y_true=y_valid, y_pred=knn_valid_pred, squared=False)
valid_rmse_rf = mean_squared_error(y_true=y_valid, y_pred=rf_valid_pred, squared=False)
print("Test RMSE using KNN:", valid_rmse_knn)
print("Test RMSE using Randomforest:", valid_rmse_rf)


# 3. save model
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="knn_model")
args = parser.parse_args()
mlflow.set_experiment("knn_rf-exp")
input_sample = X_train.iloc[:10]

## run experiment
if args.model_name == "knn_model":
    signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=knn_train_pred)
    with mlflow.start_run():
        mlflow.log_metrics({"train_rmse": train_rmse_knn, "valid_rmse": valid_rmse_knn})
        mlflow.sklearn.log_model(
            sk_model=knn_pipeline,
            artifact_path=args.model_name,
            signature=signature,
            input_example=input_sample,
        )

elif args.model_name == "rf_model":
    with mlflow.start_run():
        signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=rf_train_pred)
        mlflow.log_metrics({"train_rmse": train_rmse_rf, "valid_rmse": valid_rmse_rf})
        mlflow.sklearn.log_model(
            sk_model=rf_pipeline,
            artifact_path=args.model_name,
            signature=signature,
            input_example=input_sample,
        )

# 4. save data
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_valid, y_valid], axis=1)

df.to_csv("data.csv", index=False)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)



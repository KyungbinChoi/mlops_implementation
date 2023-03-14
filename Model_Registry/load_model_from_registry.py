from argparse import ArgumentParser
import mlflow
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

# 0. set mlflow environments
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. load model from mlflow
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="knn_model")
parser.add_argument("--run-id", dest="run_id", type=str)
args = parser.parse_args()

# 2. load model
print('Import trained model : run_id -- {}, model name -- {} '.format(args.run_id,args.model_name))
model_pipeline = mlflow.pyfunc.load_model(f"runs:/{args.run_id}/{args.model_name}")

# 3. prediction 
test_df = pd.read_csv('./test_data.csv')
X_test = test_df.drop(["medhouseval"], axis='columns')
y_test = test_df["medhouseval"]

test_pred = model_pipeline.predict(X_test)
test_rmse = mean_squared_error(y_true=y_test, y_pred=test_pred, squared=False)

print("Test RMSE :: ", test_rmse)
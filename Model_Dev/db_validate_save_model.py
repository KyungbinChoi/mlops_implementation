import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

# 1. import test data set
df = pd.read_csv("test_data.csv")
X_test = df.drop(["medhouseval"], axis="columns")
y_test = df["medhouseval"]

# 2. load model
knn_pipeline_load = joblib.load("db_pipeline_knn.joblib")
rf_pipeline_load = joblib.load("db_pipeline_rf.joblib")

# 3. validate
knn_load_valid_pred = knn_pipeline_load.predict(X_test)
rf_load_valid_pred = rf_pipeline_load.predict(X_test)

knn_load_valid_acc = mean_squared_error(y_true=y_test, y_pred=knn_load_valid_pred, squared=False)
rf_load_valid_acc = mean_squared_error(y_true=y_test, y_pred=rf_load_valid_pred, squared=False)

print("Load Model Test RMSE :", knn_load_valid_acc)
print("Load Model Test RMSE :", rf_load_valid_acc)
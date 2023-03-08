import joblib
import pandas as pd
import psycopg2
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# 1. get data
db_connect = psycopg2.connect(host="localhost", database="mydatabase", user="myuser", password="mypassword")
df = pd.read_sql("SELECT * FROM housing_data", db_connect)
X = df.drop(["id", "medhouseval"], axis="columns")
y = df["medhouseval"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

# 2. model development and train
knn_pipeline = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsRegressor(n_neighbors=3))])
knn_pipeline.fit(X_train, y_train)
knn_train_pred = knn_pipeline.predict(X_train)


rf_pipeline = Pipeline([("rf", RandomForestRegressor())])
rf_pipeline.fit(X_train, y_train)
rf_train_pred = rf_pipeline.predict(X_train)

train_acc_knn = mean_squared_error(y_true=y_train, y_pred=knn_train_pred, squared=False)
train_acc_rf = mean_squared_error(y_true=y_train, y_pred=rf_train_pred, squared=False)
print("Train RMSE using KNN:", train_acc_knn)
print("Train RMSE using Randomforest:", train_acc_rf)

# 3. save model
joblib.dump(knn_pipeline, "db_pipeline_knn.joblib")
joblib.dump(rf_pipeline, "db_pipeline_rf.joblib")

# 4. save data
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_valid, y_valid], axis=1)

df.to_csv("data.csv", index=False)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
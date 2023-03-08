import joblib
import pandas as pd
import psycopg2
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 1. get data
db_connect = psycopg2.connect(host="localhost", database="mydatabase", user="myuser", password="mypassword")
df = pd.read_sql("SELECT * FROM housing_data", db_connect)
X = df.drop(["id", "medhouseval"], axis="columns")
y = df["medhouseval"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

# 2. model define and grid search
rf_pipeline = Pipeline([("rf", RandomForestRegressor(n_jobs=-1, random_state=42))])

rf_parmas = {'rf__n_estimators':[500,700,800], 'rf__max_depth':[18,20,24]}
grid_cv = GridSearchCV(rf_pipeline, param_grid=rf_parmas, cv= 5, n_jobs=-1)
grid_cv.fit(X_train, y_train)
print('best parameters = {}'.format(grid_cv.best_params_))

rfcv_train_pred = grid_cv.predict(X_train)
rfcv_test_pred = grid_cv.predict(X_valid)

train_acc_rfcv = mean_squared_error(y_true=y_train, y_pred=rfcv_train_pred, squared=False)
tset_acc_rfcv = mean_squared_error(y_true=y_valid, y_pred=rfcv_test_pred, squared=False)
print("Train RMSE using Randomforest with grid-search:", train_acc_rfcv)
print("Test RMSE using Randomforest with grid-search:", tset_acc_rfcv)

# 3. save model
joblib.dump(grid_cv, "db_pipeline_rfcv.joblib")

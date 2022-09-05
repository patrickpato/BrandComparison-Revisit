import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import mlflow 
import mlflow.sklearn  
X = np.array([-2, -1, 0, 1,  2, 1]).reshape(-1,1)
y = np.array([0, 0, 1, 1, 1, 0])
lr = LogisticRegression()
rf = RandomForestClassifier()
mlp = MLPClassifier()
lr.fit(X, y), rf.fit(X, y), mlp.fit(X,y)

lr_score = lr.score(X, y)
rf_score = rf.score(X,y)
mlp_score = mlp.score(X,y) 
print("Score: %s", lr_score)
mlflow.log_metric("lr_score", lr_score)
mlflow.log_metric("rf_score", rf_score)
mlflow.log_metric("mlp_score", mlp_score)
mlflow.sklearn.log_model(lr, "model")
mlflow.sklearn.log_model(rf, "model")
mlflow.sklearn.log_model(mlp, "model")
print("Model saved in run  %s"% mlflow.active_run().info.run_uuid)
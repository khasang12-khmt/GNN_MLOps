from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow
from prefect import flow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error
from sklearn import datasets

mlflow.set_experiment("Test2")

@flow
def train():
    mlflow.sklearn.autolog()

    iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    #iris = datasets.load_iris()
    #x = iris.data
    #y = iris.target

    features = ["sepal_length","sepal_width", "petal_length", "petal_width"]
    target = ["species"]

    X_train, X_test, y_train, y_test = train_test_split(iris[features], iris[target].values, test_size=0.33)
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    
    params = {"max_depth": 2, "random_state": 42}
    forest = RandomForestClassifier(**params)

    with mlflow.start_run() as run:
        forest.fit(X_train, y_train)
        
        predictions = forest.predict(X_test)
        
        mlflow.sklearn.log_model(forest, "random-forest-model")

        # log model performance 
        mlflow.log_params(params)

# if __name__ == "__main__":
#     train()
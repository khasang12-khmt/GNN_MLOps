from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from mlflow import MlflowClient
import mlflow.pyfunc
import pandas as pd
app = FastAPI()


def fetch_latest_model():
    client = MlflowClient()
    return dict(client.search_registered_models()[0])["name"]


def fetch_latest_version(model_name):
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/latest"
    )
    return model


@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)


@app.get("/predict/")
def model_output(user: int, item: int):
    print("Works I")
    model_name = fetch_latest_model()
    model = fetch_latest_version(model_name)
    print("Works II")
    # Prepare the model input
    model_input = pd.DataFrame({"user_indices": [user], "item_indices": [item]})
    #model_input = [user,item]

    # Make predictions using the model
    prediction = model.predict(model_input)

    # Return the prediction as a JSON response
    return prediction

# @app.get("/predict/")
# def model_output(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
#     print("Works I")
#     model_name = fetch_latest_model()
#     model = fetch_latest_version(model_name)
#     print("Works II")
#     input = pd.DataFrame({"sepal_length": [sepal_length], "sepal_width": [sepal_width], "petal_length": [petal_length], "petal_width": [petal_width]})

#     prediction = model.predict(input)
#     print(prediction)
#     return {"prediction": prediction[0]}

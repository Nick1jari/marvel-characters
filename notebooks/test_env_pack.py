# Databricks notebook source

import json
import os

import mlflow
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel
from marvel_characters.models.custom_model import adjust_predictions
from marvel_characters.serving.model_serving import ModelServing


# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "main"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config,
                         tags=tags,
                         spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()
basic_model.train()

# COMMAND ----------
# basic model
mlflow.set_experiment(basic_model.experiment_name)
with mlflow.start_run(tags=basic_model.tags) as run:
    a_run_id = run.info.run_id

    signature = infer_signature(
        model_input=basic_model.X_train,
        model_output=basic_model.pipeline.predict(basic_model.X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=basic_model.pipeline,
        artifact_path="lightgbm-pipeline-model",
        signature=signature
    )

a_registered_model = mlflow.register_model(
        model_uri=f"runs:/{a_run_id}/lightgbm-pipeline-model",
        name=basic_model.model_name,
        tags=basic_model.tags,
        env_pack="databricks_model_serving"
    )

model_serving = ModelServing(
    model_name=basic_model.model_name,
      endpoint_name="marvel-character-basic-envpack"
)
model_serving.deploy_or_update_serving_endpoint(version=a_registered_model.version)

# COMMAND ----------
# custom model with code path

class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for LightGBM model."""

    def ___init__(self, pipeline: sklearn.pipeline.Pipeline):
        self.model = pipeline


    def predict(self, context: PythonModelContext,
                model_input: pd.DataFrame | np.ndarray) -> dict:
        """Predict the survival of a character."""
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)

code_paths = ["marvel_characters-0.1.0-py3-none-any.whl"]
mlflow.set_experiment(experiment_name=config.experiment_name_custom)
with mlflow.start_run(tags=basic_model.tags):
    b_run_id = run.info.run_id
    conda_env = _mlflow_conda_env(
        additional_pip_deps=["code/marvel_characters-0.1.0-py3-none-any.whl"])

    signature = infer_signature(
        model_input=basic_model.X_train,
        model_output={"Survival prediction": ["alive"]})
    model_info = mlflow.pyfunc.log_model(
        python_model=MarvelModelWrapper(basic_model.pipeline),
        name="pyfunc-wrapper",
        signature=signature,
        code_paths=code_paths,
        conda_env=conda_env,
    )

custom_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom"
b_registered_model = mlflow.register_model(
        model_uri=f"runs:/{b_run_id}/pyfunc-wrapper",
        name=custom_model_name,
        tags=basic_model.tags,
        env_pack="databricks_model_serving"
    )

model_serving = ModelServing(
      model_name=custom_model_name,
      endpoint_name="marvel-character-custom-envpack-codepath"
)
model_serving.deploy_or_update_serving_endpoint(version=b_registered_model.version)

# COMMAND ----------
with mlflow.start_run(tags=basic_model.tags):
    c_run_id = run.info.run_id

    signature = infer_signature(
        model_input=basic_model.X_train,
        model_output={"Survival prediction": ["alive"]})
    model_info = mlflow.pyfunc.log_model(
        python_model=MarvelModelWrapper(basic_model.pipeline),
        name="pyfunc-wrapper",
        signature=signature,
    )

custom_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom_c"
c_registered_model = mlflow.register_model(
        model_uri=f"runs:/{c_run_id}/pyfunc-wrapper",
        name=custom_model_name,
        tags=basic_model.tags,
        env_pack="databricks_model_serving"
    )

model_serving = ModelServing(
      model_name=custom_model_name,
      endpoint_name="marvel-character-custom-envpack-no-codepath"
)
model_serving.deploy_or_update_serving_endpoint(version=c_registered_model.version)

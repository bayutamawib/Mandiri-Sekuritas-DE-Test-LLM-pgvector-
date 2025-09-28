import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow

# Load model config
with open("src/churn_pipeline/config/model.yaml", "r") as f:
  config = yaml.safe_load(f)

embedding_dim = config["embedding_dim"]
numeric_features = config["numeric_features"]
dropout_rate = config["hyperparameters"]["dropout_rate"]
hidden_units = config["hyperparameters"]["hidden_units"]
learning_rate = config["hyperparameters"]["learning_rate"]
batch_size = config["hyperparameters"]["batch_size"]
epochs = config["hyperparameters"]["epochs"]

def build_model(embedding_dim, numeric_dim):
  embed_input = layers.Input(shape=(embedding_dim,), name="embedding_input")
  x1 = layers.LayersNormalization()(embed_input)
  x1 = layers.dense(hidden_units[0], activation="relu")(x1)
  x1 = layers.Dropout(dropout_rate)(x1)

  num_input = layers.Input(shape=(numeric_dim,), name="numeric_input")
  x2 = layers.BatchNormalization()(num_input)
  x2 = layers.Dense(hidden_units[1], activation="relu")(x2)

  x = layers.concatenate([x1, x2])
  x = layers.Dense(hidden_units[2], activation="relu")(x)
  x = layers.Dropout(0.2)(x)
  output = layers.Dense(1, activation="sigmoid", name="churn_score")(x)

  model = models.Model(inputs[embed_input, num_input], outputs=output)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC()])
  return model

def train(X_embed, X_num, y, version="latest", model_dir="models"):
    mlflow.set_experiment("ChurnPrediction")
    with mlflow.start_run(run_name=f"train_v{version}"):
       model = build_model(X_embed.shape[1], X_num.shape[1])
       model.fit([X_embed, X_num], y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"keras_churn_v{version}")
    model.save(model_path)
    mlflow.tensorflow.log_model(tf_saved_model_dir=model_path, artifact_path="model", artifact_path="model")

    mlflow.log_params({
       "embedding_dim": embedding_dim,
       "numeric_dim": X_num.shape[1],
       "dropout_rate": dropout_rate,
       "hidden_units": hidden_units,
       "learning_rate": learning_rate,
       "batch_size": batch_size,
       "epochs": epochs,
       "training_samples": len(y)
    })

    print(f"Model saved to {model_path}")
    return model
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def make_model(window_size, num_feats):
    tf.keras.utils.set_random_seed(42)
    in_layer = tf.keras.layers.Input(shape=(window_size, num_feats))
    std_layer = tf.keras.layers.LayerNormalization()(in_layer)

    lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)(std_layer)
    drop1 = tf.keras.layers.Dropout(0.3)(lstm1)

    lstm2 = tf.keras.layers.LSTM(64)(drop1)
    drop2 = tf.keras.layers.Dropout(0.3)(lstm2)

    dense = tf.keras.layers.Dense(32, activation="relu")(drop2)
    drop3 = tf.keras.layers.Dropout(0.2)(dense)

    out_layer = tf.keras.layers.Dense(1)(drop3)
    return tf.keras.models.Model(inputs=in_layer, outputs=out_layer)

offset = int(input("Target version: "))
v = int(input("Features' version: "))
versions = [["dollar", "sentiment_-1", "sentiment_1"],
            ["sentiment_-1", "sentiment_1"],
            ["dollar"],
            []]

data = pd.read_csv("../../assets/dataset.csv")
data = pd.concat([data["date"],
                  data.drop(["date", "sentiment"], axis=1).diff(),
                  data["sentiment"]], axis=1)
data = pd.get_dummies(data, columns=["sentiment"], dtype=int).drop("sentiment_0", axis=1)

targets = pd.read_csv("../../assets/targets.csv")

df = pd.concat([data, targets[str(offset)].rename("target")], axis=1) # Joins input and target
df.dropna(inplace=True, ignore_index=True)
df.drop(versions[v - 1], axis=1, inplace=True)
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

num_feats = len(df.columns) - 2

limit1 = datetime(2022, 1, 1) # Limit date for training
limit2 = datetime(2023, 7, 1) # Limit date for validation
train_data = df[df["date"] < limit1].drop("date", axis=1)
val_data = df[(df["date"] >= limit1) & (df["date"] < limit2)].drop("date", axis=1).reset_index(drop=True)

for window in range(10, 60 + 1, 10):
    train = tf.keras.preprocessing.timeseries_dataset_from_array(
        np.array(train_data.drop("target", axis=1)), # Input
        np.array(train_data.loc[(window - 1):, "target"]), # Target
        sequence_length=window,
        batch_size=None)
    val = tf.keras.preprocessing.timeseries_dataset_from_array(
        np.array(val_data.drop("target", axis=1)),
        np.array(val_data.loc[(window - 1):, "target"]),
        sequence_length=window,
        batch_size=None)

    train = train.shuffle(train.cardinality(), seed=42).batch(8)
    val = val.shuffle(val.cardinality(), seed=42).batch(8)

    model = make_model(window, num_feats)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"), tf.keras.metrics.R2Score(name="R2")])
    stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                restore_best_weights=True,
                                                start_from_epoch=500)
    hist = model.fit(train, epochs=600, callbacks=[stopping], validation_data=val)

    model.save(f"../../results/models/model_t+{offset}_v{v}_w{window}.keras")

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))
    ax0.plot(hist.history["MAE"], label="entrenamiento")
    ax0.plot(hist.history["val_MAE"], label="validación")
    ax0.set_title("MAE")
    ax0.set_xlabel("Época")
    ax0.legend()
    ax1.plot(hist.history["R2"], label="entrenamiento")
    ax1.plot(hist.history["val_R2"], label="validación")
    ax1.set_title("R^2")
    ax1.set_xlabel("Época")
    ax1.legend()

    fig.savefig(f"../../results/graphs/graphs_t+{offset}_v{v}_w{window}.png")

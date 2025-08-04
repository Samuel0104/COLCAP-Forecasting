from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("../../assets/dataset.csv")
data = pd.concat([data["date"],
                  np.sign(data.drop(["date", "sentiment"], axis=1).diff()),
                  data["sentiment"]], axis=1)
data = pd.get_dummies(data, columns=["sentiment"], dtype=int).drop("sentiment_0", axis=1)

targets = pd.read_csv("../../assets/targets.csv")
targets = np.sign(targets).replace(-1, 0)

limit1 = datetime(2022, 1, 1) # Limit date for training
limit2 = datetime(2023, 7, 1) # Limit date for validation

versions = [["dollar", "sentiment_-1", "sentiment_1"],
            ["sentiment_-1", "sentiment_1"],
            ["dollar"],
            []]

for offset in range(1, 3 + 1):
    acc = pd.DataFrame(columns=["version", "window", "train", "validation", "test"])
    
    for v in range(1, 4 + 1):
        df = pd.concat([data, targets[str(offset)].rename("target")], axis=1) # Joins input and target
        df.dropna(inplace=True, ignore_index=True)
        df.drop(versions[v - 1], axis=1, inplace=True)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        
        train_data = df[df["date"] < limit1].drop("date", axis=1)
        val_data = df[(df["date"] >= limit1) & (df["date"] < limit2)].drop("date", axis=1).reset_index(drop=True)
        test_data = df[df["date"] >= limit2].drop("date", axis=1).reset_index(drop=True)

        for window in range(10, 60 + 1, 10):
            train = tf.keras.preprocessing.timeseries_dataset_from_array(
                np.array(train_data.drop("target", axis=1)), # Input
                np.array(train_data.loc[(window - 1):, "target"]), # Target
                sequence_length=window,
                batch_size=1)
            val = tf.keras.preprocessing.timeseries_dataset_from_array(
                np.array(val_data.drop("target", axis=1)),
                np.array(val_data.loc[(window - 1):, "target"]),
                sequence_length=window,
                batch_size=1)
            test = tf.keras.preprocessing.timeseries_dataset_from_array(
                np.array(test_data.drop("target", axis=1)),
                np.array(test_data.loc[(window - 1):, "target"]),
                sequence_length=window,
                batch_size=1)

            model = tf.keras.models.load_model(f"../../results/simple_models/model_t+{offset}_v{v}_w{window}.keras")
            train_acc = model.evaluate(train, return_dict=True)["accuracy"]
            val_acc = model.evaluate(val, return_dict=True)["accuracy"]
            test_acc = model.evaluate(test, return_dict=True)["accuracy"]
            
            acc.loc[len(acc)] = [v, window, train_acc, val_acc, test_acc]
        
    acc.to_csv(f"../../results/simple_metrics/acc_t+{offset}.csv", index=False)

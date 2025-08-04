import pandas as pd

data = pd.read_csv("../../assets/dataset.csv")

targets = pd.DataFrame()
for offset in range(1, 3 + 1):
    targets[str(offset)] = round(data["close"].diff(offset).shift(-offset), 2)
    
targets.to_csv("../../assets/targets.csv", index=False)

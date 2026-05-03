import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

fake = fake[["text", "label"]]
true = true[["text", "label"]]

data = pd.concat([fake, true])

data = data.sample(frac=1, random_state=42)

data.to_csv("news_dataset.csv", index=False)

print("✅ news_dataset.csv created")
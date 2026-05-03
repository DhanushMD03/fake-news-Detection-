import pandas as pd
import random

real_news_samples = [
    "Government passes new economic reform bill",
    "Scientists discover new method to treat cancer",
    "Election commission announces voting dates",
    "New technology improves renewable energy efficiency",
    "University releases official research findings"
]

fake_news_samples = [
    "Aliens have landed in major cities worldwide",
    "Secret government project controls the weather",
    "Celebrity reveals shocking conspiracy theory",
    "Miracle cure discovered but hidden from public",
    "Time traveler warns about future disaster"
]

data = []

# Generate 2000 samples
for _ in range(1000):
    text = random.choice(real_news_samples) + " " + str(random.randint(1, 10000))
    data.append([text, 1])  # Real

for _ in range(1000):
    text = random.choice(fake_news_samples) + " " + str(random.randint(1, 10000))
    data.append([text, 0])  # Fake

# Shuffle
random.shuffle(data)

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "label"])

# Save
df.to_csv("news_dataset.csv", index=False)

print("Dataset generated: news_dataset.csv")
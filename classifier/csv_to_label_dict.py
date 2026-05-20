import pandas as pd

df = pd.read_csv("sunda_lemes_clean.csv")

dataset = []

for _, row in df.iterrows():

    loma = str(row["basa_loma"]).strip()
    lemes1 = str(row["basa_lemes_1"]).strip()
    lemes2 = str(row["basa_lemes_2"]).strip()

    # casual form
    dataset.append((loma, "casual"))

    # polite forms
    dataset.append((lemes1, "polite"))

    # second polite form (avoid duplicates)
    if lemes2 and lemes2 != lemes1:
        dataset.append((lemes2, "formal"))

# convert to dataframe
train_df = pd.DataFrame(dataset, columns=["sundanese", "politeness"])

print(train_df.head())
print(train_df["politeness"].value_counts())

train_df.to_csv("dataset.csv", index=False)
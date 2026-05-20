import pandas as pd
import re

def clean_text(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x in ["–", "-", "—", "", "nan", "None"]:
        return None

    x = re.sub(r"\((.*?)\)", r"\1", x)
    x = re.sub(r"\s+", " ", x)

    return x.strip()


df = pd.read_csv("dataset.csv")

print("BEFORE:", pd.read_csv("dataset.csv").shape)


df["sundanese"] = df["sundanese"].apply(clean_text)
df["politeness"] = df["politeness"].apply(clean_text)

# HARD FILTER (important)
df = df.dropna(subset=["sundanese", "politeness"])

expanded = []

for _, row in df.iterrows():
    text = row["sundanese"]

    # extra safety
    if not isinstance(text, str):
        continue

    words = text.split(",")

    for w in words:
        w = w.strip()

        # final guard
        if w and w.lower() not in ["nan", "none", "-"]:
            expanded.append([w, row["politeness"]])


df = pd.DataFrame(expanded, columns=["sundanese", "politeness"])

# FINAL CLEANING STEP (VERY IMPORTANT)
df = df[
    df["sundanese"].notna() &
    df["politeness"].notna()
]

df = df[df["sundanese"].str.strip() != ""]
df = df[df["politeness"].str.strip() != ""]

df["sundanese"] = df["sundanese"].str.replace("-", " ", regex=False)

print(df.head())
print(df.isna().sum())

df.to_csv("dataset_clean.csv", index=False)

print("AFTER:", df.shape)
print("\n===== REGISTER COUNTS =====")
print(df["politeness"].value_counts())
print("\n===== PERCENTAGES =====")
print(df["politeness"].value_counts(normalize=True) * 100)
import pandas as pd
import re
import sys
from pathlib import Path
import chardet



BASE_DIR = Path(__file__).resolve().parent

file_to_clean = sys.argv[1]

DATA_PATH = BASE_DIR.parent / "data" / f"{file_to_clean}"

with open(DATA_PATH, "rb") as f:
    raw_data = f.read(100000)  # read first 100KB is enough
    result = chardet.detect(raw_data)

input(f"Detected encoding: {result['encoding']}")
print(f"Confidence: {result['confidence']}")

encoding = result['encoding'] if result['confidence'] > 0.5 else 'utf-8'
#### removes parentheses, extra spaces, and standardizes missing values to None
#### have to check if everything is needed as it might

def clean_text(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x.lower() in ["–", "-", "—", "", "nan", "none"]:
        return None

    x = re.sub(r"\((.*?)\)", r"\1", x)  ### removes parentheses but keeps content inside
    x = re.sub(r"\s+", " ", x)   ### removes extra spaces

    return x.strip()

if len(sys.argv) <= 1:
    print("No file specified. Use: python clean_data.py dataset.csv")
    exit()



df = pd.read_csv(DATA_PATH, encoding="utf-8")  # try utf-8 first, fallback to detected encoding if it fails

## Initial stats
print("BEFORE:", df.shape)


df["sundanese"] = df["sundanese"].apply(clean_text)
df["register"] = df["register"].apply(clean_text)

# HARD FILTER (important)
df = df.dropna(subset=["sundanese", "register"])

# expanded = []  ####### this one is for cleaning dictionary style politeness datasets

# for _, row in df.iterrows():
#     text = row["sundanese"]

#     # extra safety
#     if not isinstance(text, str):
#         continue

#     words = text.split(",")

#     for w in words:
#         w = w.strip()

#         # final guard
#         if w and w.lower() not in ["nan", "none", "-"]:
#             expanded.append([w, row["register"]])


# # df = pd.DataFrame(expanded, columns=["sundanese", "register"])

# # FINAL CLEANING STEP (VERY IMPORTANT)
# df = df[
#     df["sundanese"].notna() &
#     df["register"].notna()
# ]

df = df[df["sundanese"].str.strip() != ""]
df = df[df["register"].str.strip() != ""]

df["sundanese"] = df["sundanese"].str.replace(r"\s*-\s*", " ", regex=True)

print(df.head())
print(df.isna().sum())

df.to_csv(f"{file_to_clean.split('.')[0]}_clean.csv", index=False)

print("AFTER:", df.shape)
print("\n===== REGISTER COUNTS =====")
print(df["register"].value_counts())
print("\n===== PERCENTAGES =====")
print(df["register"].value_counts(normalize=True) * 100)
print(df.sample(5))
print(df["sundanese"].apply(type).value_counts())
print(df[df["sundanese"].str.len() < 3])
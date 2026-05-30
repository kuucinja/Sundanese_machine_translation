import pandas as pd
import time
import json
import os
from deep_translator import GoogleTranslator
from pathlib import Path
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification



# -------------------------
# CONFIG
# -------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = "C:\\Users\\HPElitebook1\\Desktop\\mt_uppsala\\proj\\classifier\\data\\wikipedia_politeness.csv"
OUTPUT_FILE = "gt_outputs.csv"
CACHE_FILE = "translation_cache.json"

SLEEP_TIME = 0.3  # safe rate limit buffer

CLASSIFIER_PATH = BASE_DIR.parent / "classifier" / "scripts" / "models" / "sundanese_xlmr"

class_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
class_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
print("Loaded classifier:", CLASSIFIER_PATH)

print(class_model.config.id2label)
print(class_model.config.label2id)


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INPUT_FILE, sep=";")
df["register"] = pd.NA

translator = GoogleTranslator(source="en", target="su")

def classify_politeness(text: str) -> str:
    inputs = class_tokenizer(text, return_tensors="pt")
    outputs = class_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    return class_model.config.id2label[predicted_class]

# -------------------------
# LOAD CACHE (if exists)
# -------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}


outputs = []

for i, row in df.iterrows():
    en_text = str(row["text"])

    # ---- CACHE HIT ----
    if en_text in cache:
        su_text = translator.translate(en_text)
        register_label = classify_politeness(su_text)

        cache[en_text] = {
            "sundanese": su_text,
            "register": register_label
        }

        print(f"[OK] {i}")
        print(f"[CACHE HIT] {i}")

    # ---- CACHE MISS ----
    else:
        try:
            su_text = translator.translate(en_text)
            register_label = classify_politeness(su_text)

            cache[en_text] = {
                "sundanese": su_text,
                "register": register_label
            }

            print(f"[OK] {i}")

        except Exception as e:
            su_text = None
            register_label = None
            print(f"[ERROR] {i}: {e}")

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_TIME)

    # ---- FINAL OUTPUT ----
    outputs.append({
        "english": en_text,
        "sundanese": su_text,
        "register": register_label
    })

# -------------------------
# SAVE FINAL OUTPUT
# -------------------------
out_df = pd.DataFrame(outputs)
out_df.to_csv(OUTPUT_FILE, index=False)

print("DONE: translation complete")
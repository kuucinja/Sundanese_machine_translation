import pandas as pd
import time
import json
import os
from deep_translator import GoogleTranslator

# -------------------------
# CONFIG
# -------------------------
INPUT_FILE = "test_manually_corrected.csv"
OUTPUT_FILE = "gt_outputs.csv"
CACHE_FILE = "translation_cache.json"

SLEEP_TIME = 0.3  # safe rate limit buffer

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INPUT_FILE)

translator = GoogleTranslator(source="en", target="su")

# -------------------------
# LOAD CACHE (if exists)
# -------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

outputs = []

# -------------------------
# TRANSLATION LOOP
# -------------------------
for i, row in df.iterrows():
    en_text = str(row["english"])

    # ---- check cache first ----
    if en_text in cache:
        su_text = cache[en_text]
        print(f"[CACHE HIT] {i}")
    else:
        try:
            su_text = translator.translate(en_text)
            cache[en_text] = su_text
            print(f"[OK] {i}")
        except Exception as e:
            su_text = None
            print(f"[ERROR] {i}: {e}")

        # save cache after each request (important for recovery)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_TIME)

    outputs.append({
        "id": i,
        "english": en_text,
        "google_translate_sundanese": su_text
    })

# -------------------------
# SAVE FINAL OUTPUT
# -------------------------
out_df = pd.DataFrame(outputs)
out_df.to_csv(OUTPUT_FILE, index=False)

print("DONE: translation complete")
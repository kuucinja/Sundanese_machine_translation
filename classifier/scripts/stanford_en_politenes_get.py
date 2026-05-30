from pathlib import Path
import pandas as pd
from convokit import Corpus, download

corpus = Corpus(filename=download("wikipedia-politeness-corpus"))

data = []

for convo in corpus.iter_conversations():
    for utt in convo.iter_utterances():

        data.append({
            "id": utt.id,
            "text": utt.text,
            "speaker": utt.speaker.id if utt.speaker else None,

            # MAIN LABELS
            "binary_label": utt.meta.get("Binary"),
            "politeness_score": utt.meta.get("Normalized Score"),

            # optional extra info
            "annotations": utt.meta.get("Annotations"),
            "parsed": utt.meta.get("parsed")
        })

df = pd.DataFrame(data)
df = pd.DataFrame(data)

# -----------------------------
# SAVE TO DATA DIRECTORY
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

# create folder if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

output_path = DATA_DIR / "wikipedia_politeness.csv"

df.to_csv(output_path, index=False, encoding="utf-8")

print("Saved to:", output_path)
print("Shape:", df.shape)
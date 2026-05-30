from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import os
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CLASSIFIER_PATH = BASE_DIR.parent / "classifier" / "models" /"sundanese_xlmr" 

class_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
class_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
print("Loaded classifier:", CLASSIFIER_PATH)

# -----------------------------
# NLLB model setup
# -----------------------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(class_model.config.id2label)
print(class_model.config.label2id)


# -----------------------------
# NLLB language codes
# -----------------------------

SRC_LANG = "eng_Latn"
TGT_LANG = "sun_Latn"   # Sundanese (Latin script)


def translate(text: str) -> str:
    # Set source language
    tokenizer.src_lang = SRC_LANG

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")

    # Force target language
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    # Generate translation
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=200
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def classify_politeness(text: str) -> str:
    inputs = class_tokenizer(text, return_tensors="pt")
    outputs = class_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    return class_model.config.id2label[predicted_class]

def translate(text: str):

    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(text, return_tensors="pt")

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=200
    )

    sundanese = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'pre-classification: {sundanese}')

    # 👇 classify output
    politeness = classify_politeness(sundanese)

    return {
        "translation": sundanese,
        "politeness": politeness
    }



# Interactive translation loop

while True:
    user_input = input("\nEnter sentence (/exit to quit): ")

    # Exit condition
    if user_input.strip().lower() == "/exit":
        print("Exiting...")
        break

    # Skip empty input
    if not user_input.strip():
        continue

    try:
        result = translate(user_input)

        print("\nTranslation:", result["translation"])
        print("Politeness:", result["politeness"])

    except Exception as e:
        print("Error:", e)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# NLLB language codes
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


# Example
print(translate("How are you today?"))
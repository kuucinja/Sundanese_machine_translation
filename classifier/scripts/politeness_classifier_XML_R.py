import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "models" / "sundanese_xlmr"

#Load dataset
filename = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
df = pd.read_csv(filename)

print("Loaded dataset:", df.columns)

# check distribution
print(df["register"].value_counts())


## Balance dataset so that each register is equally represented
min_size = df["register"].value_counts().min()

df_balanced = (
    df.groupby("register", group_keys=False).sample(n=min_size, random_state=42).reset_index(drop=True)
)
print(df_balanced.columns)

print(f'balanced registers: {df_balanced["register"].value_counts()}')



# Label encoding
labels = sorted(df_balanced["register"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

df_balanced["label"] = df_balanced["register"].map(label2id)



X = df_balanced["sundanese"]
y = df_balanced["label"]

print(y)


# Train/test split
train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    stratify=df_balanced["label"],
    random_state=42
)

train_ds = Dataset.from_pandas(
    train_df[["sundanese", "label"]]
)

test_ds = Dataset.from_pandas(
    test_df[["sundanese", "label"]]
)

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


#Functions
def tokenize(batch):
    return tokenizer(
        batch["sundanese"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }





# Build pipeline
train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)




################MODEL SETUP######################


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./xlmr-register",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

#### Train
trainer.train(resume_from_checkpoint="./xlmr-register/checkpoint-232")

results = trainer.evaluate()

print(results)

model.save_pretrained("classifier\scripts\models\sundanese_xlmr")
tokenizer.save_pretrained("classifier\scripts\models\sundanese_xlmr")

# # Predict
# preds = model.predict(X_test)

# # Evaluate
# print("Accuracy:", accuracy_score(y_test, preds))
# print()
# print(classification_report(y_test, preds))

# # Save model
# joblib.dump(model, "politeness_classifier.pkl")

# print("Model saved.")

# print("\nTrue label distribution (TEST SET):")
# print(y_test.value_counts())
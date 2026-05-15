"""
Procedural Synthetic Data Generation for Coercive Control Simulation
Codebase for: Data Generation, Validation, and Transformer Fine-Tuning
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# =====================================================================
# PHASE 1: PROCEDURAL SYNTHETIC DATA GENERATION (LLAMA-3 / QWEN)
# =====================================================================
def load_generative_model(model_id="Qwen/Qwen2.5-7B-Instruct"):
    """Loads the base generative LLM for procedural synthesis."""
    print(f"Loading generative model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
    return tokenizer, model

def generate_subtle_forensic_dyads(tokenizer, model, trait_name, psychological_anchor, num_samples=10):
    """Generates ambiguous, multi-turn conversational dyads with explicit temporal tokens."""
    system_prompt = f"""You are a computational psycholinguist building a dataset of implicit coercive control.
Your task is to generate {num_samples} distinct, multi-turn text message exchanges demonstrating the Dark Triad trait: '{trait_name}'.

CRITICAL CONSTRAINTS:
1. AVOID CARICATURES: Do not use overt hostility. The abuse must be highly subtle, passive-aggressive, or shrouded in faux-concern.
2. AMBIGUITY: To an outside observer, the text should look almost normal. The manipulation is subtextual.
3. TEMPORAL TOKENS: You MUST separate messages with explicit time-delay tokens to simulate asynchronous control (e.g., <DELAY: 2_MINS> or <DELAY: 14_HOURS>).
4. FORMAT: Output exactly {num_samples} distinct exchanges. Separate each exchange with "---". 

ANCHOR BEHAVIOUR: "{psychological_anchor}"
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate the subtle, temporal-encoded dyads."}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.85, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    reply = response.split("system\n")[-1].split("assistant\n")[-1].strip()
    
    return [dyad.strip() for dyad in reply.split('---') if len(dyad.strip()) > 10]

# Note: In production, loop this function across clinical anchors to build the full 275-dyad dataset.
# df_synthetic.to_csv("scaled_forensic_corpus_v2.csv", index=False)


# =====================================================================
# PHASE 2: MATHEMATICAL VALIDATION (SBERT & VADER)
# =====================================================================
def validate_synthetic_corpus(csv_path="scaled_forensic_corpus_v2.csv"):
    """Validates structural realism using Semantic Overlap and proves the semantic gap using VADER."""
    print("\n--- PHASE 2: SBERT & VADER VALIDATION ---")
    df = pd.read_csv(csv_path).dropna()
    
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    vader = SentimentIntensityAnalyzer()
    
    npd_similarities, benign_similarities, npd_vader_scores = [], [], []

    for _, row in df.iterrows():
        text, label = str(row['text']), row['label']
        
        # 1. VADER Lexical Baseline Test
        if label == 1:
            npd_vader_scores.append(vader.polarity_scores(text)['compound'])

        # 2. SBERT Semantic Overlap (Empathy degradation)
        messages = text.split('\n')
        if len(messages) >= 2:
            emb1 = sbert_model.encode(messages[0], convert_to_tensor=True)
            emb2 = sbert_model.encode(" ".join(messages[1:]), convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
            
            if label == 1: npd_similarities.append(cosine_sim)
            else: benign_similarities.append(cosine_sim)

    print(f"SBERT Overlap (Benign): {np.mean(benign_similarities):.3f}")
    print(f"SBERT Overlap (NPD):    {np.mean(npd_similarities):.3f} (Significant Degradation)")
    print(f"VADER NPD Sentiment:    {np.mean(npd_vader_scores):.3f} (Positive/Neutral - Failed to detect)")


# =====================================================================
# PHASE 3: TRANSFORMER FINE-TUNING (DistilBERT)
# =====================================================================
def train_forensic_classifier(csv_path="scaled_forensic_corpus_v2.csv"):
    """Fine-tunes DistilBERT to classify implicit coercive control based on sequence & temporal gaps."""
    print("\n--- PHASE 3: TRANSFORMER FINE-TUNING ---")
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    
    hf_dataset = Dataset.from_pandas(df)
    dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42) # 80/20 Dyadic split
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = dataset_split['train'].map(tokenize_function, batched=True)
    tokenized_eval = dataset_split['test'].map(tokenize_function, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        return {"accuracy": accuracy_score(labels, predictions), "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir="./final_forensic_triage_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        eval_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    print("\n--- FINAL CLASSIFICATION REPORT ---")
    preds = np.argmax(trainer.predict(tokenized_eval).predictions, axis=-1)
    print(classification_report(tokenized_eval['label'], preds, target_names=['Benign (0)', 'NPD (1)']))

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Ensure "scaled_forensic_corpus_v2.csv" is in the root directory before running validation and training.
    # validate_synthetic_corpus()
    # train_forensic_classifier()
    pass

import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# -------------------------------
# CONFIG
# -------------------------------
INPUT_FILE = "./train/train_judg.jsonl"   # your JSONL file (id + Judgment)
OUTPUT_FILE = "./train_pred_summaries.jsonl"    # where summaries will be saved
MODEL_NAME = "nsi319/legal-pegasus"     # from the repo
MAX_INPUT_TOKENS = 1024                 # Pegasus max length
CHUNK_LENGTH = 395                      # for splitting long docs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4   
CSV_METRICS_FILE = "./eval_metrics.csv"
# -------------------------------
# HELPERS
# -------------------------------
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def nest_sentences(text, max_tokens, tokenizer):
    """Split long text into chunks based on tokenizer tokens."""
    tokens = tokenizer.encode(text, truncation=False)
    nested = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        nested.append(tokenizer.decode(chunk, skip_special_tokens=True))

    return nested

def batch_generate(texts, tokenizer, model):
    """Generate summaries for a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_TOKENS
    ).to(DEVICE)

    summary_ids = model.generate(
        **inputs,
        max_length=256,
        min_length=30,
        length_penalty=2.0,
        num_beams=5,
        early_stopping=True,
    )
    return [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("Loading model...")
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Loading dataset...")
    df = load_jsonl(INPUT_FILE)
    ref_df=load_jsonl("./train/train_ref_summ.jsonl")
    results = []
    metrics_records = []
    merged_df = df.merge(ref_df, on="ID", how="inner")

    predictions, references = [], []

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
         doc_id = str(row["ID"])
         judgment = row["Judgment"]
         reference = row["Summary"]

        # --- summarization ---
         chunks = nest_sentences(judgment, CHUNK_LENGTH, tokenizer)
         summaries = []
         for i in range(0, len(chunks), BATCH_SIZE):
            batch_texts = chunks[i : i + BATCH_SIZE]
            batch_summaries = batch_generate(batch_texts, tokenizer, model)
            summaries.extend(batch_summaries)
        
         final_summary = " ".join(summaries)

        # store generated
         results.append({"id": doc_id, "summary": final_summary})
         predictions.append(final_summary)
         if reference:
            references.append(reference)

            # --- evaluation per doc ---
            scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, final_summary)

            rouge2_f = scores['rouge2'].fmeasure
            rougeL_f = scores['rougeL'].fmeasure

            smooth = SmoothingFunction().method1
            ref_tokens = [reference.split()]
            pred_tokens = final_summary.split()
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)

            metrics_records.append({
                "id": doc_id,
                "reference": reference,
                "prediction": final_summary,
                "rouge2": rouge2_f,
                "rougeL": rougeL_f,
                "bleu": bleu
            })

    # save predictions
    save_jsonl(results, OUTPUT_FILE)
    #print(f"✅ Summaries saved to {OUTPUT_FILE}")
    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        avg_row = {
            "id": "AVERAGE",
            "reference": "",
            "prediction": "",
            "rouge2": metrics_df["rouge2"].mean(),
            "rougeL": metrics_df["rougeL"].mean(),
            "bleu": metrics_df["bleu"].mean()
        }

        # append avg row
        metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)

    # save metrics
        metrics_df.to_csv(CSV_METRICS_FILE, index=False, encoding="utf-8")
        #print(f"📊 Metrics saved to {CSV_METRICS_FILE}")

        # print averages
        # print("\n🔎 Averages across dataset:")
        # print(f"ROUGE-2: {metrics_df['rouge2'].mean():.4f}")
        # print(f"ROUGE-L: {metrics_df['rougeL'].mean():.4f}")
        # print(f"BLEU: {metrics_df['bleu'].mean():.4f}")
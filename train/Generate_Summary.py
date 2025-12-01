import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import spacy
nlp = spacy.load("en_core_web_sm")


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


def batch_generate(texts, tokenizer, model, max_input_tokens, device):
    """Generate summaries for a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_input_tokens
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=256,
        min_length=20,
        length_penalty=2.0,
        repetition_penalty=2.5,
        num_beams=5,
        early_stopping=True,
    )
    return [tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Pegasus summarization on a dataset.")

    parser.add_argument("--input", required=True,
                        help="Path to input test file (.jsonl or .csv)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file to save summaries")
    parser.add_argument("--model_dir", required=True,
                        help="Path to fine-tuned Pegasus model directory")
    parser.add_argument("--base_model", default="nsi319/legal-pegasus",
                        help="Base model name or path")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for summarization")
    parser.add_argument("--max_input_tokens", type=int, default=1024,
                        help="Max input tokens for Pegasus")

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading fine-tuned model...")
    tokenizer = PegasusTokenizer.from_pretrained(args.base_model)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_dir).to(DEVICE)

    print("Loading test dataset...")

    # Load JSONL or CSV automatically
    if args.input.endswith(".jsonl"):
        df = load_jsonl(args.input)
    else:
        df = pd.read_csv(args.input)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(row["ID"])
        judgment = row["para_text"]

        summaries = []
        for i in range(0, len(judgment), args.batch_size):
            batch_texts = judgment[i : i + args.batch_size]
            batch_summaries = batch_generate(
                batch_texts,
                tokenizer,
                model,
                args.max_input_tokens,
                DEVICE
            )
            summaries.extend(batch_summaries)

        raw_summary = " ".join(summaries)

        # Store prediction
        results.append({
            "ID": doc_id,
            "Summary": raw_summary
        })

    save_jsonl(results, args.output)
    print(f"Summaries saved to {args.output}")

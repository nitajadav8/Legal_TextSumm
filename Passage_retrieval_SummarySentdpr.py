import json
import pandas as pd
import argparse
from transformers import BertTokenizer
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import os
import math
import numpy as np
import re

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Process text and extract DPR training data.")

parser.add_argument("--judgment", required=True, help="Path to input judgment JSONL file")
parser.add_argument("--summary", required=True, help="Path to input summary JSONL file")
parser.add_argument("--output", required=True, help="Path to output JSONL")

args = parser.parse_args()

# -------------------------
# Model + Tokenizer Loading
# -------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

if torch.cuda.is_available():
    print("cuda device available")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

# -------------------------
# Helper Functions
# -------------------------
tokens_for_sentence = {}

def tokenize(text):
    if text not in tokens_for_sentence:
        tokens_for_sentence[text] = tokenizer.tokenize(text)
    return tokens_for_sentence[text]

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def save_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def estimate_parts(tokens, max_length=600):
    return max(1, -(-len(tokens) // max_length))

def get_sentence_from_token(sentence_tokens):
    token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    return tokenizer.decode(token_ids)

def split_paragraph(paragraph, max_length=600, last=False):
    sentences = split_into_sentences(paragraph)
    all_tokens = [token for sentence in sentences for token in tokenize(sentence)]

    if len(all_tokens) < 256 and not last:
        return None

    num_parts = estimate_parts(all_tokens, max_length)

    parts = []
    current_part_tokens = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)

        if current_token_count + len(sentence_tokens) > (len(all_tokens) // num_parts):
            parts.append(current_part_tokens)
            current_part_tokens = []
            current_part_tokens.append(get_sentence_from_token(sentence_tokens))
            current_token_count = len(sentence_tokens)
        else:
            current_part_tokens.append(get_sentence_from_token(sentence_tokens))
            current_token_count += len(sentence_tokens)

    if current_part_tokens and (not parts or parts[-1] != current_part_tokens):
        parts.append(current_part_tokens)

    return parts

def process_text(text):
    paragraphs = text.split('\n')
    processed = []
    last = False

    for idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        if idx == len(paragraphs) - 1:
            last = True

        processed_paragraph = split_paragraph(paragraph, 600, last)
        if processed_paragraph is None:
            paragraphs[idx + 1] = paragraph + "\n" + paragraphs[idx + 1]
            continue

        processed.extend(processed_paragraph)

    return processed

def clean_processed_text(p_list):
    print("cleaning")
    cleaned = []

    for p in p_list:
        if not isinstance(p, (list, tuple)):
            continue

        cleaned_para = []
        for s in p:
            if s is None:
                continue
            if isinstance(s, float) and (math.isnan(s) or np.isnan(s)):
                continue
            s = str(s).strip()
            if s.lower() in {"nan", "none", "null", ""}:
                continue
            s = re.sub(r'\s+', ' ', s)
            s = s.replace('.,', '.')
            cleaned_para.append(s)

        if cleaned_para:
            cleaned.append(cleaned_para)

    return cleaned

def encode_texts(model, texts):
    return model.encode(texts)

def find_most_relevant_passage_sentence_level(model, input_sentence, passages):
    print("matching")
    sentence_embedding = encode_texts(model, [input_sentence])

    highest = second = third = -1
    idx1 = idx2 = idx3 = -1

    seen = []

    for i, passage in enumerate(passages):
        if passage in seen:
            print("----- duplicate -----")
            continue
        seen.append(passage)

        emb = encode_texts(model, passage)
        sim = cosine_similarity(sentence_embedding, emb).max()

        if sim > highest:
            third, second, highest = second, highest, sim
            idx3, idx2, idx1 = idx2, idx1, i
        elif sim > second:
            third, second = second, sim
            idx3, idx2 = idx2, i
        elif sim > third:
            third = sim
            idx3 = i

    return {
        "highest_similarity": highest,
        "second_highest_similarity": second,
        "third_highest_similarity": third,
        "most_relevant_passage": " ".join(passages[idx1]),
        "second_most_relevant_passage": " ".join(passages[idx2]),
        "third_most_relevant_passage": " ".join(passages[idx3]),
    }

# -------------------------
# Load Input Files
# -------------------------
judg = load_jsonl(args.judgment)
summ = load_jsonl(args.summary)

merged_df = judg.merge(summ, on="ID", how="inner")

# -------------------------
# Process
# -------------------------
results = []

for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    ID = str(row["ID"])
    judgment = row["newJudgement"]
    Summary = row["Summary"]

    passages = process_text(judgment)
    passage_list = clean_processed_text(passages)
    summary_sents = split_into_sentences(Summary)

    for sentence in summary_sents:
        sim_dict = find_most_relevant_passage_sentence_level(model, sentence, passage_list)
        rec = {"ID": ID, "Summary_sentence": sentence}
        rec.update(sim_dict)
        results.append(rec)



new_df = pd.DataFrame(results)
save_jsonl(new_df, args.output)

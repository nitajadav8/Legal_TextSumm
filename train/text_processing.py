import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import sys
from nltk import tokenize
import torch
import nltk
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download('punkt_tab')

# -------------------------------------------------
# Command-line argument parser
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Chunk documents and generate training data.")
parser.add_argument("--input", required=True, help="Path to input JSONL file")
parser.add_argument("--output", required=True, help="Path to output CSV file")
args = parser.parse_args()

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
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

def split_to_sentences_summ(para):
    return tokenize.sent_tokenize(para)

def nest_sentences_update(document, chunk_length=1024):
    nested = []
    sent = []
    length = 0
    
    for sentence in nltk.sent_tokenize(document):
        sent_len = len(sentence.split(" "))
        
        if sent_len >= chunk_length:
            if sent:
                nested.append(sent)
                sent = []
                length = 0
            nested.append([sentence])
            continue
        
        if length + sent_len <= chunk_length:
            sent.append(sentence)
            length += sent_len
        else:
            nested.append(sent)
            sent = [sentence]
            length = sent_len
      
    if sent:
        nested.append(sent)
    return nested

def similarity_l_l(l1, l2):
    document_embeddings = sbert_model.encode(l1 + l2)
    similarities = cosine_similarity(document_embeddings)
    result = []
    for i in range(len(l1)):
        vals = similarities[i]
        vals = vals[len(l1):]
        idx = np.argmax(vals)
        result.append(idx)
    return result

def get_chunks_data_from_docV2(doc, summ):
    chunk_summ_word_threshold = 150
    sentence_mapping = {}

    doc_sents = split_to_sentences_summ(doc)
    summ_sents = split_to_sentences_summ(summ)

    result = similarity_l_l(summ_sents, doc_sents)

    for i in range(len(summ_sents)):
        sentence_mapping[doc_sents[result[i]]] = summ_sents[i]
    
    final_chunks = []
    final_summ = []

    for chunk in nest_sentences_update(doc, 1024):
        s = ""
        for chunk_sent in chunk:
            if chunk_sent in sentence_mapping:
                s += sentence_mapping[chunk_sent]

        if len(s.split(" ")) >= chunk_summ_word_threshold:
            final_chunks.append(" ".join(chunk))
            final_summ.append(s)

    return final_chunks, final_summ

# -------------------------------------------------
# Load SBERT
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)

# -------------------------------------------------
# Load input file
# -------------------------------------------------
df = load_jsonl(args.input)

# -------------------------------------------------
# Process data
# -------------------------------------------------
training_chunks = []
training_summs = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    doc_id = str(row["ID"])
    judgment = row["Judgment"]
    reference = row["Summary"]

    cks, summs = get_chunks_data_from_docV2(judgment, reference)

    training_chunks += cks
    training_summs += summs

# -------------------------------------------------
# Save results
# -------------------------------------------------
full = list(zip(training_chunks, training_summs))
tdf = pd.DataFrame(full, columns=['data', 'summary'])

tdf.to_csv(args.output, index=False)

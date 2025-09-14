import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import sys
from nltk import tokenize
import torch
import nltk
nltk.download('punkt_tab')

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
    sents = tokenize.sent_tokenize(para)
    return sents


def nest_sentences_update(document, chunk_length=1024):
    """
    Function to chunk a document into chunks of ~chunk_length words
    without exceeding the limit (except if a single sentence is longer).
    
    input:
        document (str)       - Input document text
        chunk_length (int)   - Maximum number of words per chunk (default 1024)
    
    output:
        nested (list) - List of chunks, each chunk is a list of sentences
    """
    nested = []
    sent = []
    length = 0
    
    for sentence in nltk.sent_tokenize(document):
        sent_len = len(sentence.split(" "))
        
        # If a single sentence is longer than chunk_length, force it into its own chunk
        if sent_len >= chunk_length:
            if sent:  # save the current chunk first
                nested.append(sent)
                sent = []
                length = 0
            nested.append([sentence])  # sentence alone forms a chunk
            continue
        
        # If adding this sentence stays within the limit
        if length + sent_len <= chunk_length:
            sent.append(sentence)
            length += sent_len
        else:
            # Save the current chunk, start a new one with this sentence
            nested.append(sent)
            sent = [sentence]
            length = sent_len
      
    if sent:
        nested.append(sent)
    return nested

def similarity_l_l(l1, l2):
    '''
    Function to find the most similar sentence in the document for each sentence in the summary 
    input:  l1 - Summary sentences
            l2 - Document sentences
    returns a list of document sentence indexes for each sentence in the summary 
    '''
    document_embeddings = sbert_model.encode(l1+l2)
    similarities=cosine_similarity(document_embeddings)
    result = []
    for i in range(len(l1)):
        vals = similarities[i]
        vals = vals[len(l1):]
        idx = np.argmax(vals)
        result.append(idx)
    return result

def get_chunks_data_from_docV2(doc, summ):
    '''
    Function to generate chunks along with their summaries 
    input:  doc - legal Document
            summ - Gold standard summary
    returns a list of chunks and their summaries 
    '''
    chunk_summ_word_threshold = 150
    sentence_mapping = {}
    doc_sents = split_to_sentences_summ(doc)
    summ_sents = split_to_sentences_summ(summ)
    
    result = (similarity_l_l(summ_sents,doc_sents))
    
    for i in range(len(summ_sents)):
        sentence_mapping[doc_sents[result[i]]] = summ_sents[i]
    
    final_chunks = []
    final_summ = []
    for chunk in nest_sentences_update(doc, 1024):
        summ = ""
        for chunk_sent in chunk:
            if chunk_sent in sentence_mapping:
                summ = summ + sentence_mapping[chunk_sent]
        if len(summ.split(" ")) >= chunk_summ_word_threshold:
            final_chunks.append(" ".join(chunk))
            final_summ.append(summ)
    return final_chunks, final_summ


# Loading Model and tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)
df = load_jsonl('./dataset/train/train_split.jsonl')
training_chunks = []
training_summs = []
for _, row in tqdm(df.iterrows(), total=len(df)):
         doc_id = str(row["ID"])
         judgment = row["Judgment"]
         reference = row["Summary"]
         cks, summs = get_chunks_data_from_docV2(judgment,reference)
         training_chunks = training_chunks + cks
         training_summs = training_summs + summs        
full = list(zip(training_chunks,training_summs))
tdf = pd.DataFrame(full,columns=['data', 'summary']) 
df.to_csv("TrainFD_LPMCS_512.csv", index=False)

import json
import pandas as pd
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

if torch.cuda.is_available():
    print("cuda device available")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)



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
    """
    Splits the text into sentences using spaCy's sentence boundary detection.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def estimate_parts(tokens, max_length=600):
    """
    Estimate the number of parts needed based on the total token count.
    """
    return max(1, -(-len(tokens) // max_length))

def get_sentence_from_token(sentence_tokens):
    token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    decoded_text = tokenizer.decode(token_ids)
    return decoded_text

def split_paragraph(paragraph, max_length=600, last=False):
    """
    Split a paragraph into multiple parts, each as close to equal length as possible,
    without exceeding max_length tokens, and breaking at sentence ends.
    """
    sentences = split_into_sentences(paragraph)
    all_tokens = [token for sentence in sentences for token in tokenize(sentence)]
    if(len(all_tokens)<256 and not last):
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
    # Add the last part if it's not empty
    if current_part_tokens and (not parts or parts[-1] != current_part_tokens):
        parts.append(current_part_tokens)
    
    return parts

def process_text(text):
    """
    Process the entire text, splitting it into paragraphs and further splitting each paragraph.
    """
    paragraphs = text.split('\n')
    processed_paragraphs = []
    last = False
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        if index == len(paragraphs) - 1:
            last = True
        processed_paragraph = split_paragraph(paragraph,600,last)
        if processed_paragraph is None:
            paragraphs[index+1] = paragraph + "\n" + paragraphs[index+1]
            continue
        processed_paragraphs.extend(processed_paragraph)
    return processed_paragraphs

def clean_processed_text(p_list):
    print("cleaning")
    cleaned = []
    for p in p_list:
        if not isinstance(p, (list, tuple)):
            continue  

        cleaned_paragraph = []
        for s in p:
            if s is None:
                continue
            if isinstance(s, float) and (math.isnan(s) or np.isnan(s)):
                continue
            s = str(s).strip()
            if s.lower() in {"nan", "none", "null", ""}:
                continue
            if len(s) < 1:
                continue
            s = re.sub(r'\s+', ' ', s)  
            s = s.replace('.,', '.')    
            s = s.strip()
            cleaned_paragraph.append(s)

        if cleaned_paragraph:
            cleaned.append(cleaned_paragraph)

    return cleaned

def encode_texts(model, texts):
    """
    Encode a list of texts using the provided model.

    :param model: Loaded model.
    :param texts: List of texts (sentences or passages) to encode.
    :return: List of encoded embeddings.
    """
    return model.encode(texts)

def find_most_relevant_passage_sentence_level(model, input_sentence, passages):
    """
    Find the most relevant passage for the given sentence, comparing at the sentence level.

    :param model: Loaded model.
    :param input_sentence: Input sentence for which to find relevant passage.
    :param passages: List of passages, each being a list of sentences.
    :return: Most relevant passage.
    """
    print("matching")
    sentence_embedding = encode_texts(model, [input_sentence])

    highest_similarity = -1
    second_highest_similarity = -1  # Initialize to a low value
    third_highest_similarity = -1
    
    
    most_relevant_passage_index = -1
    second_most_relevant_passage_index = -1
    third_most_relevant_passage_index = -1
    
    all_passages = []
    
    # Iterate over each passage
    for i, passage in enumerate(passages):
        if passage in all_passages:
            print("------------------duplicate-----------------")
            continue
        all_passages.append(passage)
        
        # Encode each sentence in the passage
        passage_embeddings = encode_texts(model, passage)

        # Calculate similarities for each sentence in the passage
        similarities = cosine_similarity(sentence_embedding, passage_embeddings)

        # Find the highest similarity score in this passage
        max_similarity_in_passage = similarities.max()
        # Check if this passage contains the most similar sentence so far
        if max_similarity_in_passage > highest_similarity:
            third_highest_similarity = second_highest_similarity
            second_highest_similarity = highest_similarity
            highest_similarity = max_similarity_in_passage
    
            third_most_relevant_passage_index = second_most_relevant_passage_index
            second_most_relevant_passage_index = most_relevant_passage_index
            most_relevant_passage_index = i
            
        elif max_similarity_in_passage > second_highest_similarity:
            third_highest_similarity = second_highest_similarity
            second_highest_similarity = max_similarity_in_passage
            third_most_relevant_passage_index = second_most_relevant_passage_index
            second_most_relevant_passage_index = i
            
        elif max_similarity_in_passage > third_highest_similarity:
            third_highest_similarity = max_similarity_in_passage
            third_most_relevant_passage_index = i
    similarity_dict = {
        'highest_similarity': highest_similarity,
        'second_highest_similarity': second_highest_similarity,
        'third_highest_similarity': third_highest_similarity,
        'most_relevant_passage': " ".join(passages[most_relevant_passage_index]),
        'second_most_relevant_passage': " ".join(passages[second_most_relevant_passage_index]),
        'third_most_relevant_passage': " ".join(passages[third_most_relevant_passage_index]),
    }
    return similarity_dict


judg=load_jsonl('train/train_gpt_cleaned_4.jsonl')
summ=load_jsonl('train/train_ref_summ.jsonl')
output_file='dpr3_train_data_4.jsonl'
merged_df=judg.merge(summ, on="ID", how="inner")
'''
#Resuming#
processed_ids = set()
if os.path.exists(output_file):
    print(f"Resuming from {output_file} ...")
    with open(output_file, 'r',encoding="utf-8") as reader:
        for row in reader:
            processed_ids.add((row["ID"], row["Summary"]))
else:
    print("Starting fresh...")

with open(output_file,'w', encoding="utf-8") as writer:
 '''
results=[]
for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
     ID= str(row["ID"])
     judgment = row["newJudgement"]
     Summary = row["Summary"]
     passages=process_text(judgment)
     passage_list=clean_processed_text(passages)
     summary_sents = split_into_sentences(Summary)

     for sentence in summary_sents:
         #if (ID, sentence) in processed_ids:
          #   continue  
         sim_dict = find_most_relevant_passage_sentence_level(model, sentence, passage_list)
         tosave = {"ID": ID, "Summary_sentence": sentence}
         tosave.update(sim_dict)
         results.append(tosave) 
     break
new_df = pd.DataFrame(results)
save_jsonl(new_df,output_file)

    

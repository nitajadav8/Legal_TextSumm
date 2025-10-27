from transformers import  AutoTokenizer
from transformers import LEDTokenizer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-led-base-16384") #tokenizer model
import spacy
nlp = spacy.load("en_core_web_sm")

device='cuda' if torch.cuda.is_available() else 'cpu'
sbert_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)


def split_to_sentences_summ(para):
     sents=[]
     if isinstance(para, (list, tuple)):
        new_sent=" ".join(map(str, para))
        doc = nlp(new_sent)
     elif isinstance(para, str):
        doc = nlp(para)
        for sent in doc.sents:
            s=sent.text.strip()
            sents.append(s)
     return sents

def count_tokens(text):
    return len(tokenizer(text, truncation=False)["input_ids"])

def cosine_sim(A, b):
    """A: (n,d), b: (d,) -> (n,)"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(A, axis=1) * (np.linalg.norm(b) + 1e-10))
    denom[denom == 0] = 1e-10
    return (A @ b) / denom

def cosine_sim_array(A, B):
    # A: (n,d) B: (m,d) -> returns (n,m)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T

def mmr_for_query(query_emb, candidate_embs, top_k=2, lambda_param=0.7):
    # query_emb: (d,), candidate_embs: (n,d)
    relevance = cosine_sim(candidate_embs, query_emb)  # (n,)
    # candidate-to-candidate sim
    cand_sim = cosine_sim_array(candidate_embs, candidate_embs)  # (n,n)
    selected = []
    if candidate_embs.shape[0] == 0:
        return selected
    selected.append(int(np.argmax(relevance)))
    while len(selected) < min(top_k, candidate_embs.shape[0]):
        remaining = [i for i in range(candidate_embs.shape[0]) if i not in selected]
        mmr_scores = []
        for r in remaining:
            max_sim = max(cand_sim[r, s] for s in selected) if selected else 0.0
            score = lambda_param * relevance[r] - (1 - lambda_param) * max_sim
            mmr_scores.append((r, score))
        next_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(int(next_idx))
    return selected

def semantic_chunk_sentences(sentences, sent_embs, token_word_limit=700, sim_threshold=0.2):
    """
    Merge contiguous sentences if similarity to last sentence in current chunk >= sim_threshold
    and chunk_word_count + next_word_count <= token_word_limit.
    (This is a simple heuristic — you can improve it using sentence transformers + heuristics.)
    """
    chunks = []
    chunk_indices = []
    cur_chunk = [sentences[0]]
    cur_idx = [0]
    cur_words = count_tokens(sentences[0])
    print("first:",cur_words)
    for i in range(1, len(sentences)):
        sim = cosine_sim_array(sent_embs[i:i+1], sent_embs[cur_idx[-1]:cur_idx[-1]+1])[0,0]
        next_words = count_tokens(sentences[i])
       
        if sim >= sim_threshold and (cur_words + next_words) <= token_word_limit:
            cur_chunk.append(sentences[i])
            cur_idx.append(i)
            cur_words += next_words
        else:
            chunks.append(" ".join(cur_chunk))
            chunk_indices.append(list(cur_idx))
            cur_chunk = [sentences[i]]
            cur_idx = [i]
            cur_words = next_words
    # finalize
    if cur_chunk:
        chunks.append(" ".join(cur_chunk))
        chunk_indices.append(list(cur_idx))

    return chunks, chunk_indices

def get_chunks_with_mmr(doc, gold_summary,model_name='sentence-transformers/bert-base-nli-mean-tokens',top_k=2, mmr_lambda=0.7,sim_threshold=0.2,chunk_word_limit=700, min_summary_words=30):
    # 1) sentences
    doc_sents = split_to_sentences_summ(doc)
    summ_sents = split_to_sentences_summ(gold_summary)
    # 2) embeddings
    sbert_model = SentenceTransformer(model_name).to(device)
    doc_embs = sbert_model.encode(doc_sents, convert_to_numpy=True)
    summ_embs = sbert_model.encode(summ_sents, convert_to_numpy=True)
    # 3) map each summary sentence to top-k doc sentences via MMR
    mapping_doc_to_summ = {}  # doc_idx -> set(summ_idx)
    for i, q in enumerate(summ_embs):
        sel = mmr_for_query(q, doc_embs, top_k=top_k, lambda_param=mmr_lambda)
        for di in sel:
            mapping_doc_to_summ.setdefault(di, set()).add(i)
    # 4) semantic chunking
    chunks, chunk_indices = semantic_chunk_sentences(doc_sents, doc_embs, token_word_limit=chunk_word_limit, sim_threshold=sim_threshold)
    # 5) aggregate summary sentences per chunk
    final_chunks, final_summaries = [], []
    for idx_list in chunk_indices:
        mapped = []
        for di in idx_list:
            if di in mapping_doc_to_summ:
                mapped.extend(sorted(list(mapping_doc_to_summ[di])))
        # deduplicate preserving order
        seen = set(); collected = []
        for midx in mapped:
            if midx not in seen:
                collected.append(midx); seen.add(midx)
        merged_summary = " ".join([summ_sents[m] for m in collected])
        if len(merged_summary.split()) >= min_summary_words:
            final_chunks.append(" ".join([doc_sents[i] for i in idx_list]))
            final_summaries.append(merged_summary)

    return final_chunks, final_summaries

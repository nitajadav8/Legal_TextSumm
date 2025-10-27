from groq import Groq
import pandas as pd
client= Groq(api_key="")

def split_into_paragraphs(doc):
    prompt = f"""
You are a text cleaning assistant.
Take the following legal document text and split it into clean paragraphs.
- Remove empty lines.
- Remove meaningless statements.
- Expand abbreviations accroding to Indian legal context.
- Remove numbering like "1.", "(i)", "(a)" from the beginning of line.
- Merge broken lines into proper sentences.
- Keep meaningful paragraph breaks.
- Correct spelling of misspelled or noisy english words containing digits or any non-alphabatic characters in between the alphabets.
- Return the result as a list of paragraph containing maximum 5 sentences.

Document:
{doc}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content



def summarize(doc):
    prompt = f"""
You are a abstractive text summarizer for legal document in Indian legal domain
-Generate a coherence summary from the given document.
-Return the result as a list of maximum 3 consistent sentences.
Document:
{doc}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

val=pd.read_csv('val_para.csv')
rows = []
for _, row in val.iterrows():
    doc_id = row["doc_id"]
    doc=row["para_text"]
    para = summarize(doc)
    for i, c in enumerate(para):
        rows.append({"doc_id": doc_id, "para_id": i, "para_text": doc, "summary":c})



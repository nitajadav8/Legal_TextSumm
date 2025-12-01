from groq import Groq
import pandas as pd
import argparse
client= Groq(api_key="") """api key here"""


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

def main():
    parser = argparse.ArgumentParser(description="Summarize paragraph CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV ")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    val = pd.read_csv(args.input)

    rows = []
    for _, row in val.iterrows():
        doc_id = row["doc_id"]
        doc = row["para_text"]
        summaries = summarize(doc)

        for i, s in enumerate(summaries):
            rows.append({
                "doc_id": doc_id,
                "para_id": i,
                "para_text": doc,
                "summary": s
            })

    out_df = pd.DataFrame(rows)

    merged = out_df.groupby("doc_id")["summary"].apply(list).reset_index()

    merged.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()



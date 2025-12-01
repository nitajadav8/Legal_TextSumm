# Legal_TextSumm

This is repository containing code that helps to generate summary from raw judgment documents. The sample data of legal document is available in the train and validation folder in json format. The data is shared by JustNLP workshop for the shared task.
## Get started (Requirements and Setup)
Python version >= 3.6
Install PyTorch (CUDA 12.1 build):
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```
Install core libraries:
```bash
pip install transformers pandas tqdm
pip install sentence-transformers
pip install spacy
pip install bitsandbytes
```
Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```
## How to Run the Code

To get the summary using LLM models
```bash
python Doc_Summary_LLM.py --input val_para.csv --output summaries.csv
```
To Normalize raw legal document incorporating legal domain knowledge using LLM

```bash
python Text_normalization_LLM.py --input raw_para.csv --output final_para.csv
```
Passage Retrieval

```bash
#using DPR
python Passage_retrieval_SummarySentdpr.py \
    --judgment train/nromalized_para.jsonl \
    --summary train/train_ref_summ.jsonl \
    --output dpr3_train_data_4.jsonl
#using MMR
python Passage_retrieval_SummarySentdpr.py \
    --judgment train/nromalized_para.jsonl \
    --summary train/train_ref_summ.jsonl \
    --output mmr_train_data_4.jsonl
```




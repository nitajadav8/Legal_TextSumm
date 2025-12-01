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






# Legal_TextSumm

This is repository containing code that helps to generate summary from raw judgment documents. The sample data of legal document is available in the train and validation folder in json format. The data is shared by JustNLP workshop for the shared task.
You may require following packages to reproduce results
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers pandas tqdm
pip install sentence-transformers
pip install spacy
python -m spacy download en_core_web_sm
pip install bitsandbytes

Install PyTorch (CUDA 12.1 build):

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121


Install core libraries:

pip install transformers pandas tqdm


Install Sentence Transformers:

pip install sentence-transformers


Install spaCy:

pip install spacy


Download the spaCy English model:

python -m spacy download en_core_web_sm


Install bitsandbytes (for 8-bit/4-bit quantization):

pip install bitsandbytes



cd olmes
pip install -e .
pip install -e .[gpu]

# Install transformers locally
cd ..
cd transformers
pip install -e .

cd ..
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
pip install --upgrade pip "setuptools<70.0.0" wheel
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -U xformers==0.0.27.post2
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install vllm==0.6.1.post1
pip install datasets
pip install matplotlib
pip install h5py

# Install transformers locally
cd transformers
pip install -e .

cd ..
pip install -r requirements.txt
python -m nltk.downloader punkt
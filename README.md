#### 1. Clone the repository:
```bash

```
#### 2. Create a virtual environment and activate it:
```bash
# Make sure you have python3 and pip installed
python3 -m venv .venv
source .venv/bin/activate

# To  deactivate the environment run:
# deactivate
```
#### 3. Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
#### 4. Run a local Ollama instance
Ollama allows you to run open-source large language models, such as Llama 2, locally.
```bash
# Install Ollama on linux
curl https://ollama.ai/install.sh | sh

# Fetch the model
ollama pull llama2

# Serve 
ollama serve
```
More details here: 
https://github.com/jmorganca/ollama


To stop ollama:
```bash
pgrep ollama
>123
sudo kill 123
```

### 5. Summarize a pdf
1. Run the api

```bash
python3 serve.py
```
2. Try summarize request on: localhost:8000/docs


3. Evaluate
```bash
python3 evaluation.py
```
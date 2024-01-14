from rouge import Rouge
from datasets import load_dataset
from utils import get_model, get_number_of_tokens
from utils import get_chain
from langchain_community.document_loaders import TextLoader
import os

# Load dataset
billsum = load_dataset('billsum', split='ca_test')
texts = billsum['text'][:10]
reference_summaries = billsum['summary'][:10]
generated_summaries = []



def create_text_files(texts): 
    for i, text in enumerate(texts):
        with open(f'pdfs/{i}.txt', 'w') as f:
            f.write(text)


# Get model
context_window, llm = get_model('llama2')
# Run summarization
for i, text in enumerate(texts):
    text=text.replace('\n', '')
    # Create text file
    with open(f'pdfs/{i}.txt', 'w') as f:
        f.write(text)
    # Load document
    loader = TextLoader(f'pdfs/{i}.txt') # Assuming that text loader and pdf loader are the same
    docs = loader.load_and_split()
    # Delete text file
    os.remove(f'pdfs/{i}.txt')
    num_tokens = get_number_of_tokens(docs, llm)

    print(num_tokens)
    print(context_window)
    chain = get_chain(num_tokens, context_window, llm)
    
    try:
        generated_summaries.append(chain.invoke(docs)['output_text'])
    except Exception as e:
        print("The code failed since it won't be able to fit the documents into the LLM context length: ", e)

    

# Initialize rouge
rouge = Rouge()
# Calculate rouge scores
scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

print(scores)


#
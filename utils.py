from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import os

def get_model(model_name="llama2"):
    '''
    Returns the context window and the language model.
    '''
    # Using cpu because when using gpu it gives an error (nvidia gtx 1050 ti)
    llm = Ollama(model=model_name, num_ctx=4096, num_gpu=1)


    # TODO: Get context window from model
    context_window = 4096
    return context_window, llm

def load_documents(pdfs_folder_path='pdfs/'):
    '''
    Loads pdf documents from a folder path. 
    '''
    loader = PyPDFDirectoryLoader(pdfs_folder_path)
    docs = loader.load()
    return docs 

def get_number_of_tokens(docs, llm):
    '''
    Returns the number of tokens in the documents.
    '''
    num_tokens = 0
    for doc in docs:
        num_tokens += llm.get_num_tokens(doc.page_content)
    return num_tokens


def get_stuff_chain(llm):
    '''
    Summarizes the documents using 'Stuff'.
    '''
    # Define the prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    return chain


def get_map_reduce_chain(llm):
    '''
    Summarizes the documents using 'Map-Reduce'.
    '''
    # Map chain: map each document to an individual summary
    map_prompt_template = """
                        Write a summary of this chunk of text that includes the main points and any important details.
                        {text}
                        """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])


    # Reduce chain
    reduce_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in bullet points which covers the key points of the text.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """

    reduce_prompt = PromptTemplate(
        template=reduce_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=reduce_prompt)

    return chain

def get_chain(num_tokens, context_window, llm):
    '''
    Returns the appropriate chain based on the number of tokens.
    '''
    if num_tokens < context_window:
        print('Using stuff')
        chain = get_stuff_chain(llm)
    else:
        print('Using map-reduce')
        chain = get_map_reduce_chain(llm)

    return chain


def load_docs(filename='file', file_content=''):
    '''
    Loads a pdf into a list of Document objects
    '''
    file_path = os.path.join('pdfs', filename)
    with open(file_path, 'wb') as f:
        f.write(file_content)
    loader = PyPDFLoader(file_path)
    # Load and split
    doc = loader.load_and_split()
    # Remove files from disk
    os.remove(file_path)

    return doc


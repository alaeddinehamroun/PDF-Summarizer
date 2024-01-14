from fastapi import FastAPI
from utils import get_chain, get_model, get_number_of_tokens, load_docs
from fastapi import FastAPI, File, UploadFile



app = FastAPI(
    title = 'Langchain summarizer server',
    version='1.0',
    description='A simple API server'

)





@app.post('/summarize')
async def summarize(file: UploadFile = File(...), keep_file = False):
    '''
    Summarizes the uploaded pdf file.
    '''

    # Check if file is pdf
    if file.content_type != 'application/pdf':
        return 'Please upload a pdf file'


    # Check if file is corrupted
    # Load document
    try:
        content = await file.read()
        filename = file.filename
        docs = load_docs(filename, content)
    except Exception as e:
        return 'The file is corrupted'

    
    # Get model
    context_window, llm = get_model('llama2')

    num_tokens = get_number_of_tokens(docs, llm)

    chain = get_chain(num_tokens, context_window, llm)
    
    try:
        summary = chain.invoke(docs)['output_text']
    except Exception as e:
        print("The code failed since it won't be able to fit the documents into the LLM context length: ", e)

    return summary

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
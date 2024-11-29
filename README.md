# haystack-education

A generative document intelligence WebApp utilising a basic Haystack RAG pipeline.

The system is set up to load in all Python3 documents, downloaded from https://docs.python.org/3/. Users can then ask natural language questions about these documents (bascially, help with any aspect of Python programming!)

## Requirements:

Python3.8 (higher versions cause Haystack dependency issues)

Your OpenAI API key as an enironment variable: OPENAI_API_KEY

Note that you can easily switch out the OpenAI generator ```OpenAIGenerator(model="gpt-4o-mini")``` with something else (e.g. ```HuggingFaceLocalGenerator()```)

## Running the app

Make sure you have all dependencies installed:
```
pip install flask
pip install haystack-ai
pip install markdown-it-py mdit_plain
pip install pypdf
pip install "sentence-transformers>=3.0.0"
```

Run the app
```python app.py```

Open http://127.0.0.1:5000/ and enter your question. E.g. "Explain Python annotations in 50 words"

By default, the app will only load a couple of the converted txt documents (those in ```data/processed/small_batch```), as it takes a fair amount of time when starting the app to load the full set of Python documents. If you'd like to work with the full set simply change the path to ```data/processed```. You can also load in the PDFs, although Haystack will take time to transform all the PDFs to text files before loading. To do this, update the path to ```data/raw``` or ```data/raw/small_batch```.
See [haystack_setup.py](/haystack_setup.py#L58)

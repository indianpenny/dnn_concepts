"""
Created on Saturday Sept 21 14:37:20 2023

@author: Srishti
"""

import openai
import os
openai.api_key = "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"


from pathlib import Path
from llama_index import download_loader

PandasExcelReader = download_loader("PandasExcelReader")

loader = PandasExcelReader(pandas_config={"header": 0})
documents = loader.load_data(file=Path('./data/my_company_products.xlsx'))


from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from langchain import OpenAI
from types import FunctionType
from llama_index import ServiceContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader, load_index_from_storage
import sys
import os
import time
#documents = SimpleDirectoryReader("./data").load_data()
index = GPTVectorStoreIndex.from_documents(documents)
#index.save_to_disk("generative_index.json")
query_engine = index.as_query_engine()
while True:
	query = input("Ask a question: ")
	if not query:
		print("Goodbye")
		
	result = query_engine.query(query)
	print(result)
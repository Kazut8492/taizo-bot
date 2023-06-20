import os
import time

import openai
from dotenv import load_dotenv
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

source_name = input("Enter which source_data you want to index: ")
file_list = os.listdir("./source_data")
if source_name not in file_list:
    print("Invalid source_data name")
    exit()

start_time = time.time()

if source_name == "taizo_pdf":
    loader = DirectoryLoader(
        f"./source_data/{source_name}/", glob="./taizo.pdf", loader_cls=UnstructuredPDFLoader
    )
else:
    loader = DirectoryLoader(
        f"./source_data/{source_name}/", glob="./*.txt", loader_cls=TextLoader
    )

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n\n")

texts = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

persist_directory = f"./langchain_storage/{source_name}"
vectordb = FAISS.from_documents(texts, embeddings)
vectordb.save_local(persist_directory)

end_time = time.time()

print(f"Running time: {end_time - start_time}")

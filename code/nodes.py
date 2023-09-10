import pandas as pd
import logging
import torch
from typing import List
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import SKLearnVectorStore
log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def loadDocuments() -> PyPDFDirectoryLoader:
  log.info("Starting PDFLoader Node.")

  loader = PyPDFDirectoryLoader("./data/01_raw")
  docs = loader.load()
  log.info(f"Total pages loaded: {len(docs)}")
  return docs

def textSplitting(docs: PyPDFDirectoryLoader) -> List:
  log.info('Splitting Text into Chunks...')
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
  texts = text_splitter.split_documents(docs)
  log.info(f"Total Chunks created: {len(texts)}")
  return texts

def createEmbeddings(texts: List):
  log.info(f'Loading Embedding Model on {DEVICE}...')

  embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": DEVICE}
  )

  log.info('Saving Persistent Vector Database')
  db = SKLearnVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_path="./data/05_model_input/vectordb.parquet",
    serializer="parquet"
    )
  db.persist()

  #db = Chroma.from_documents(texts, embeddings, persist_directory="./data/#05_model_input")       
  log.info('Ingestion process completed for all pdf files.')
  return

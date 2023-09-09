import pandas as pd
import logging
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
log = logging.getLogger(__name__)

def loadDocuments() -> PyPDFDirectoryLoader:
  log.info("Starting PDFLoader Node.")

  loader = PyPDFDirectoryLoader("./data/01_raw")
  docs = loader.load()
  log.info(f"Total pages loaded: {len(docs)}")
  return docs

def textSplitting(docs: PyPDFDirectoryLoader) -> RecursiveCharacterTextSplitter:
  log.info('Splitting Text into Chunks...')
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
  texts = text_splitter.split_documents(docs)
  log.info(f"Total Chunks created: {len(texts)}")
  return texts

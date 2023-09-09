import logging
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma, SKLearnVectorStore

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def ingestTextFolder() -> ParquetDataSet:
    log = logging.getLogger(__name__)
    log.info('Starting to IngestPDFs from Raw Folder...')
    loader = PyPDFDirectoryLoader("pdfs")
    docs = loader.load()
    log.info(f"Total pages loaded: {len(docs)}")

    log.info('Spliting Text into Chunks...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    log.info(f"Total Chunks created: {len(texts)}")
        
    log.info('Loading Llama Embedding Model...')

    embeddings = HuggingFaceInstructEmbeddings(
      model_name="hkunlp/instructor-large",
      model_kwargs={"device": DEVICE}
    )
    
    log.info('Saving Persistent Vector Database')
    db = SKLearnVectorStore.from_documents(
      documents=texts,
      embeddings=embeddings,
      persist_path="./data/05_model_input/vectordb.parquet",
      serializer="parquet"
      )
    db.persist()

    #db = Chroma.from_documents(texts, embeddings, persist_directory="./data/#05_model_input")       
    log.info('Ingestion process completed for all pdf files.')
    return

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "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"


# PDF 문서 로딩
loader = PyPDFLoader("../manuals/avante_manual.pdf")
docs = loader.load()

# 문서 쪼개기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(docs)

# OpenAI 임베딩
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 벡터 저장
db = FAISS.from_documents(split_docs, embedding_model)
db.save_local("../embeddings/vehicle_manual_index")
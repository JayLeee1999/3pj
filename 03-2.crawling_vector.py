from dotenv import load_dotenv
import os
import json
import glob
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# 환경 변수 로드
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")


def load_latest_current_issues():
    """최신 현재이슈 JSON 파일을 자동으로 로드"""
    json_files = glob.glob("data2/*_BigKinds_current_issues.json")
    if not json_files:
        raise FileNotFoundError("현재이슈 JSON 파일을 찾을 수 없습니다.")
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📂 로드하는 파일: {latest_file}")
    
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    issues = data["issues"]
    docs = [
        Document(
            page_content=f"{item['제목']}\n{item['내용']}",
            metadata={"이슈번호": item["이슈번호"], "제목": item["제목"]}
        )
        for item in issues
    ]
    
    return docs


def main():
    # 문서 로드
    docs = load_latest_current_issues()
    
    # embedding 모델 객체 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Pinecone 클라이언트 초기화
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, 
        chunk_overlap=45,
        length_function=len,  # 문자수   
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 문서를 분할
    chunks = text_splitter.split_documents(docs)
    
    return chunks, embeddings, pc


if __name__ == "__main__":
    chunks, embeddings, pc = main()
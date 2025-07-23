from dotenv import load_dotenv
import os
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# 환경변수 로드
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

# CSV 파일 로드 (경로 확인 및 오류 처리)
csv_file_path = 'data/산업DB.v.0.3.csv'

# 파일 존재 여부 확인
if not os.path.exists(csv_file_path):
    print(f"파일을 찾을 수 없습니다: {csv_file_path}")
    print("현재 디렉토리:", os.getcwd())
    print("파일 경로를 확인해주세요.")
    exit(1)

try:
    loader = CSVLoader(csv_file_path, encoding='utf-8')
    docs = loader.load()
    print(f"CSV 파일 로드 완료: {len(docs)}개 문서")
except Exception as e:
    print(f"CSV 파일 로드 중 오류 발생: {e}")
    # UTF-8로 안되면 다른 인코딩 시도
    try:
        loader = CSVLoader(csv_file_path, encoding='cp949')
        docs = loader.load()
        print(f"CSV 파일 로드 완료 (cp949 인코딩): {len(docs)}개 문서")
    except Exception as e2:
        print(f"인코딩 변경 후에도 오류 발생: {e2}")
        exit(1)

# 임베딩 모델 객체 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone 클라이언트 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2300, 
    chunk_overlap=230,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# 문서 분할
chunks = text_splitter.split_documents(docs)

# 벡터 스토어에 배치로 저장
batch_size = 200

for i in range(0, len(chunks), batch_size):
    batch_docs = chunks[i:i+batch_size]
    
    if i == 0:
        # 첫 번째 배치로 벡터 스토어 생성
        vector_store = PineconeVectorStore.from_documents(
            batch_docs,
            embedding=embeddings,
            index_name="lastproject",
            namespace="industry"
        )
    else:
        # 이후 배치는 기존 벡터 스토어에 추가
        vector_store.add_documents(batch_docs)
    
    print(f"배치 {i//batch_size + 1} 완료: {len(batch_docs)}개 문서 업로드")

print("모든 문서 업로드 완료!")
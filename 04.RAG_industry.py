# industry_rag.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경변수 로드
load_dotenv(override=True)

# 2. 설정값
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lastproject")
NAMESPACE = "industry"
NEWS_JSON_PATH = "data2/2025.07.23_12.15.39_BigKinds_current_issues.json"
INDUSTRY_DB_PATH = "data/산업DB.v.0.3.csv"

# 3. 벡터 임베딩 및 벡터스토어
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding,
    namespace=NAMESPACE
)

# 4. 산업 DB 로딩
df = pd.read_csv(INDUSTRY_DB_PATH)
industry_dict = dict(zip(df["KRX 업종명"], df["상세내용"]))
valid_krx_names = set(df["KRX 업종명"].unique())

# 5. 뉴스 이슈 로딩
with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 모든 이슈에 대해 분석 수행
for idx, issue in enumerate(data["issues"]):
    print(f"\n{'='*80}")
    print(f"📰 이슈 {idx+1}/10: {issue['제목']}")
    print(f"{'='*80}")
    
    query = f"{issue['제목']}\n{issue['내용']}"

    # 6. 유사 산업 벡터 검색 (유사도 점수 포함)
    results = vector_store.similarity_search_with_score(query, k=5)

    # 7. 벡터 검색 결과에서 상세내용 추출
    matched_industries = []
    for doc, score in results:
        content = doc.page_content
        
        # BOM 문자 제거
        content = content.replace('\ufeff', '').replace('﻿', '')
        
        # 산업명과 상세내용 추출
        industry_name = None
        industry_detail = None
        
        if "KRX 업종명:" in content and "상세내용:" in content:
            lines = content.split("\n")
            
            # 산업명 추출
            for line in lines:
                if "KRX 업종명:" in line:
                    industry_name = line.replace("KRX 업종명:", "").strip()
                    break
            
            # 상세내용 추출 (벡터에서 가져온 실제 설명)
            content_parts = content.split("상세내용:")
            if len(content_parts) > 1:
                industry_detail = content_parts[1].strip()
        
        # 유효한 산업명인지 확인 및 중복 방지
        if industry_name and industry_name in valid_krx_names:
            # 이미 추가된 산업인지 확인
            already_added = any(item['name'] == industry_name for item in matched_industries)
            if not already_added:
                # 유사도를 백분율로 변환 (거리가 작을수록 유사도가 높음)
                similarity_percentage = round((1 - score) * 100, 1)
                matched_industries.append({
                    "name": industry_name, 
                    "description": industry_detail or industry_dict.get(industry_name, "설명 없음"),
                    "similarity": similarity_percentage
                })

    print(f"최종 매칭된 산업 수: {len(matched_industries)}")

    # 8. 매칭된 산업이 없을 경우 처리
    if not matched_industries:
        print("❌ 매칭된 산업이 없습니다.")
        continue

    # 9. GPT 입력 구성 (상위 3개만)
    top_industries = matched_industries[:3]
    industry_list_text = "\n".join([f"{i+1}. {item['name']} (유사도: {item['similarity']}%)" for i, item in enumerate(top_industries)])
    industry_reason_text = "\n\n".join([f"**{item['name']}** (유사도: {item['similarity']}%)\n{item['description']}" for item in top_industries])

    # 10. GPT 분석 프롬프트 (근거 포함)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 한국 산업 뉴스를 분석하는 전문 리서치 애널리스트야. 
    중요: 반드시 제공된 산업명만 사용하고, 제공된 산업 설명을 근거로 관련성을 설명해야 해."""),
        ("human", """
    [이슈 내용]
    {news}

    [관련 산업명 (유사도 포함)]
    {industry_list}

    [각 산업의 상세 설명 (근거자료)]
    {industry_reasons}

    위 정보를 바탕으로 다음 형식으로 답변해주세요:

    **이슈 요약**
    (1-2문장으로 간략히)

    **관련 산업 분석**
    - **산업명1** (유사도: X%): (제공된 산업 설명을 근거로 왜 이 이슈와 관련이 있는지 설명)
    - **산업명2** (유사도: X%): (제공된 산업 설명을 근거로 왜 이 이슈와 관련이 있는지 설명)  
    - **산업명3** (유사도: X%): (제공된 산업 설명을 근거로 왜 이 이슈와 관련이 있는지 설명)

    **주의: 반드시 위에 제공된 산업명과 설명만 사용하세요.**
    """)
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser

    # 11. GPT 호출 및 결과 출력
    response = chain.invoke({
        "news": query,
        "industry_list": industry_list_text,
        "industry_reasons": industry_reason_text
    })

    print(response)
    
    # 각 이슈 사이에 구분선 추가
    if idx < len(data["issues"]) - 1:
        print(f"\n{'-'*80}")

print(f"\n🎉 총 {len(data['issues'])}개 이슈 분석 완료!")
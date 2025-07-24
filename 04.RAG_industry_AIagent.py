# 04.RAG_industry.py

import os
import json
import pandas as pd
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 1. 환경변수 로드
load_dotenv(override=True)

# 2. 설정값
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lastproject")
NAMESPACE = "industry"
INDUSTRY_DB_PATH = "data/산업DB.v.0.3.csv"

# 3. 최신 뉴스 파일 자동 선택 함수
def get_latest_current_issues_file():
    """최신 current_issues JSON 파일 경로를 자동으로 찾기"""
    json_files = glob.glob("data2/*_BigKinds_current_issues.json")
    if not json_files:
        raise FileNotFoundError("현재이슈 JSON 파일을 찾을 수 없습니다. data2/ 폴더를 확인해주세요.")
    
    # 파일 생성 시간을 기준으로 가장 최신 파일 선택
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📂 자동 선택된 파일: {latest_file}")
    return latest_file

# 4. 파일 경로 설정 (함수 호출로 실제 파일 경로 얻기)
try:
    NEWS_JSON_PATH = get_latest_current_issues_file()
except FileNotFoundError as e:
    print(f"❌ 오류: {e}")
    print("data2/ 폴더에 *_BigKinds_current_issues.json 파일이 있는지 확인해주세요.")
    exit(1)

# 5. 벡터 임베딩 및 벡터스토어
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding,
    namespace=NAMESPACE
)

# 6. 산업 DB 로딩
df = pd.read_csv(INDUSTRY_DB_PATH)
industry_dict = dict(zip(df["KRX 업종명"], df["상세내용"]))
valid_krx_names = list(df["KRX 업종명"].unique())

# 7. AI Agent 1: 관련 산업 후보 추출
def extract_candidate_industries(news_content, industry_list, top_k=10):
    """AI Agent가 뉴스 내용을 보고 관련 가능성이 높은 산업들을 추출"""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 뉴스와 산업의 관련성을 판단하는 전문 애널리스트야.
주어진 뉴스 내용을 분석하고, 제공된 KRX 업종 리스트에서 관련 가능성이 높은 산업들을 선별해야 해.

관련성 판단 기준:
1. 직접적 영향: 뉴스가 해당 산업에 직접적인 영향을 미치는가?
2. 공급망 관계: 뉴스 관련 기업/산업과 공급망 관계가 있는가?
3. 시장 동향: 뉴스가 해당 산업의 시장 동향에 영향을 미치는가?
4. 정책/규제: 뉴스가 해당 산업 관련 정책이나 규제와 연관되는가?"""),
        ("human", """
[뉴스 내용]
{news}

[KRX 업종 리스트]
{industries}

위 뉴스와 관련 가능성이 높은 산업을 {top_k}개 선별해주세요.
각 산업에 대해 관련성 점수(1-10점)와 간단한 이유를 제시해주세요.

출력 형식 (JSON):
{{
  "candidates": [
    {{"industry": "산업명", "score": 점수, "reason": "관련성 이유"}},
    ...
  ]
}}""")
    ])
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "news": news_content,
        "industries": ", ".join(industry_list),
        "top_k": top_k
    })
    
    return result["candidates"]

# 8. AI Agent 2: 벡터 검색 결과와 AI 후보 결합
def combine_and_validate_results(news_content, vector_candidates, ai_candidates, industry_dict):
    """벡터 검색 결과와 AI 후보를 결합하여 최종 관련 산업 도출"""
    
    # 벡터 후보와 AI 후보 결합
    all_candidates = {}
    
    # 벡터 검색 결과 추가
    for candidate in vector_candidates:
        name = candidate["name"]
        all_candidates[name] = {
            "name": name,
            "vector_similarity": candidate["similarity"],
            "ai_score": 0,
            "ai_reason": "",
            "description": candidate["description"]
        }
    
    # AI 후보 추가/업데이트
    for candidate in ai_candidates:
        name = candidate["industry"]
        if name in all_candidates:
            all_candidates[name]["ai_score"] = candidate["score"]
            all_candidates[name]["ai_reason"] = candidate["reason"]
        elif name in industry_dict:  # 유효한 산업명인 경우만
            all_candidates[name] = {
                "name": name,
                "vector_similarity": 0,
                "ai_score": candidate["score"],
                "ai_reason": candidate["reason"],
                "description": industry_dict[name]
            }
    
    # 종합 점수 계산 (벡터 유사도 + AI 점수)
    for candidate in all_candidates.values():
        # 벡터 유사도를 10점 만점으로 정규화
        normalized_vector = candidate["vector_similarity"] / 10
        # AI 점수는 이미 10점 만점
        ai_score = candidate["ai_score"]
        
        # 가중평균 (AI 점수에 더 높은 가중치)
        candidate["final_score"] = round((normalized_vector * 0.3 + ai_score * 0.7), 1)
    
    # 최종 점수로 정렬
    sorted_candidates = sorted(all_candidates.values(), 
                              key=lambda x: x["final_score"], 
                              reverse=True)
    
    return sorted_candidates[:3]  # 상위 3개만 반환

# 9. 메인 실행 부분
def main():
    # 뉴스 이슈 로딩 및 분석
    with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, issue in enumerate(data["issues"]):
        print(f"\n{'='*80}")
        print(f"📰 이슈 {idx+1}/10: {issue['제목']}")
        print(f"{'='*80}")
        
        query = f"{issue['제목']}\n{issue['내용']}"

        # Step 1: 벡터 검색으로 후보 추출
        results = vector_store.similarity_search_with_score(query, k=10)
        
        vector_candidates = []
        for doc, score in results:
            content = doc.page_content.replace('\ufeff', '').replace('﻿', '')
            
            if "KRX 업종명:" in content:
                lines = content.split("\n")
                for line in lines:
                    if "KRX 업종명:" in line:
                        industry_name = line.replace("KRX 업종명:", "").strip()
                        if industry_name in industry_dict:
                            # 중복 체크
                            if not any(c["name"] == industry_name for c in vector_candidates):
                                similarity_percentage = round((1 - score) * 100, 1)
                                
                                content_parts = content.split("상세내용:")
                                industry_detail = content_parts[1].strip() if len(content_parts) > 1 else industry_dict[industry_name]
                                
                                vector_candidates.append({
                                    "name": industry_name,
                                    "similarity": similarity_percentage,
                                    "description": industry_detail
                                })
                        break

        # Step 2: AI Agent로 관련 산업 후보 추출
        print("🤖 AI Agent가 관련 산업을 분석 중...")
        ai_candidates = extract_candidate_industries(query, valid_krx_names, top_k=10)
        
        # Step 3: 결과 결합 및 검증
        final_candidates = combine_and_validate_results(query, vector_candidates, ai_candidates, industry_dict)
        
        if not final_candidates:
            print("❌ 관련 산업을 찾을 수 없습니다.")
            continue
        
        # Step 4: 최종 분석 결과 생성
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 산업 뉴스 분석 전문가야. 정확하고 신뢰성 있는 분석을 제공해야 해."),
            ("human", """
[이슈 내용]
{news}

[선별된 관련 산업]
{industries}

위 정보를 바탕으로 다음 형식으로 분석해주세요:

**이슈 요약**
(핵심 내용 1-2문장)

**관련 산업 분석**
- **산업명1** (신뢰도: X점): 관련성 설명
- **산업명2** (신뢰도: X점): 관련성 설명  
- **산업명3** (신뢰도: X점): 관련성 설명

**분석 신뢰도**: 전체적인 분석의 신뢰도를 평가해주세요.
""")
        ])
        
        industries_text = "\n".join([
            f"- {c['name']} (종합점수: {c['final_score']}/10, 벡터유사도: {c['vector_similarity']}%, AI점수: {c['ai_score']}/10)\n"
            f"  AI 판단 근거: {c['ai_reason']}\n"
            f"  산업 설명: {c['description'][:100]}..."
            for c in final_candidates
        ])
        
        parser = StrOutputParser()
        analysis_chain = analysis_prompt | llm | parser
        
        response = analysis_chain.invoke({
            "news": query,
            "industries": industries_text
        })
        
        print(response)
        
        # 디버깅 정보
        print(f"\n📊 상세 점수:")
        for i, c in enumerate(final_candidates, 1):
            print(f"{i}. {c['name']}: 종합{c['final_score']}/10 (벡터 유사도 {c['vector_similarity']}% + AI 분석 점수 {c['ai_score']}/10)")
        
        if idx < len(data["issues"]) - 1:
            print(f"\n{'-'*80}")

    print(f"\n🎉 총 {len(data['issues'])}개 이슈 분석 완료!")

if __name__ == "__main__":
    main()
# past_issue_rag.py

import os
import json
import pandas as pd
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
NAMESPACE = "past_issue"
NEWS_JSON_PATH = "data2/2025.07.23_12.15.39_BigKinds_current_issues.json"
PAST_ISSUE_DB_PATH = "data/Past_news.csv"

# 3. 벡터 임베딩 및 벡터스토어
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding,
    namespace=NAMESPACE
)

# 4. 과거 이슈 DB 로딩
df = pd.read_csv(PAST_ISSUE_DB_PATH)
issue_dict = dict(zip(df["Issue_name"], df["Contents"] + "\n\n상세: " + df["Contentes(Spec)"]))
valid_issue_names = list(df["Issue_name"].unique())

# 5. AI Agent 1: 관련 과거 이슈 후보 추출
def extract_candidate_past_issues(news_content, issue_list, top_k=10):
    """AI Agent가 뉴스 내용을 보고 관련 가능성이 높은 과거 이슈들을 추출"""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 현재 뉴스와 과거 이슈의 관련성을 판단하는 전문 애널리스트야.
주어진 현재 뉴스 내용을 분석하고, 제공된 과거 이슈 리스트에서 관련 가능성이 높은 이슈들을 선별해야 해.

관련성 판단 기준:
1. 유사한 시장 상황: 과거 이슈와 현재 상황이 유사한 시장 환경인가?
2. 동일한 산업/기업 영향: 같은 산업이나 유사한 기업들에 영향을 미치는가?
3. 정책/경제적 유사성: 정책 변화나 경제적 요인이 유사한가?
4. 투자자 심리: 투자자들의 반응이나 시장 심리가 비슷한가?"""),
        ("human", """
[현재 뉴스 내용]
{news}

[과거 이슈 리스트]
{issues}

위 현재 뉴스와 관련 가능성이 높은 과거 이슈를 {top_k}개 선별해주세요.
각 과거 이슈에 대해 관련성 점수(1-10점)와 간단한 이유를 제시해주세요.

출력 형식 (JSON):
{{
  "candidates": [
    {{"issue": "이슈명", "score": 점수, "reason": "관련성 이유"}},
    ...
  ]
}}""")
    ])
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "news": news_content,
        "issues": ", ".join(issue_list),
        "top_k": top_k
    })
    
    return result["candidates"]

# 6. AI Agent 2: 벡터 검색 결과와 AI 후보 결합
def combine_and_validate_results(news_content, vector_candidates, ai_candidates, issue_dict):
    """벡터 검색 결과와 AI 후보를 결합하여 최종 관련 과거 이슈 도출"""
    
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
        name = candidate["issue"]
        if name in all_candidates:
            all_candidates[name]["ai_score"] = candidate["score"]
            all_candidates[name]["ai_reason"] = candidate["reason"]
        elif name in issue_dict:  # 유효한 이슈명인 경우만
            all_candidates[name] = {
                "name": name,
                "vector_similarity": 0,
                "ai_score": candidate["score"],
                "ai_reason": candidate["reason"],
                "description": issue_dict[name]
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

# 7. 뉴스 이슈 로딩 및 분석
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
        
        if "Issue_name:" in content:
            lines = content.split("\n")
            for line in lines:
                if "Issue_name:" in line:
                    issue_name = line.replace("Issue_name:", "").strip()
                    if issue_name in issue_dict:
                        # 중복 체크
                        if not any(c["name"] == issue_name for c in vector_candidates):
                            similarity_percentage = round((1 - score) * 100, 1)
                            
                            content_parts = content.split("Contents:")
                            issue_detail = content_parts[1].strip() if len(content_parts) > 1 else issue_dict[issue_name]
                            
                            vector_candidates.append({
                                "name": issue_name,
                                "similarity": similarity_percentage,
                                "description": issue_detail
                            })
                    break

    # Step 2: AI Agent로 관련 과거 이슈 후보 추출
    print("🤖 AI Agent가 관련 과거 이슈를 분석 중...")
    ai_candidates = extract_candidate_past_issues(query, valid_issue_names, top_k=10)
    
    # Step 3: 결과 결합 및 검증
    final_candidates = combine_and_validate_results(query, vector_candidates, ai_candidates, issue_dict)
    
    if not final_candidates:
        print("❌ 관련 과거 이슈를 찾을 수 없습니다.")
        continue
    
    # Step 4: 최종 분석 결과 생성
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 과거 이슈와 현재 뉴스의 연관성을 분석하는 전문가야. 정확하고 신뢰성 있는 분석을 제공해야 해."),
        ("human", """
[현재 이슈 내용]
{news}

[선별된 관련 과거 이슈]
{past_issues}

위 정보를 바탕으로 다음 형식으로 분석해주세요:

**현재 이슈 요약**
(핵심 내용 1-2문장)

**관련 과거 이슈 분석**
- **과거이슈1** (신뢰도: X점): 현재 상황과의 유사점과 차이점 설명
- **과거이슈2** (신뢰도: X점): 현재 상황과의 유사점과 차이점 설명  
- **과거이슈3** (신뢰도: X점): 현재 상황과의 유사점과 차이점 설명

**시사점**: 과거 사례를 통해 예상되는 시장 반응이나 투자 전략
""")
    ])
    
    past_issues_text = "\n".join([
        f"- {c['name']} (종합점수: {c['final_score']}/10, 벡터유사도: {c['vector_similarity']}%, AI점수: {c['ai_score']}/10)\n"
        f"  AI 판단 근거: {c['ai_reason']}\n"
        f"  과거 이슈 내용: {c['description'][:200]}..."
        for c in final_candidates
    ])
    
    parser = StrOutputParser()
    analysis_chain = analysis_prompt | llm | parser
    
    response = analysis_chain.invoke({
        "news": query,
        "past_issues": past_issues_text
    })
    
    print(response)
    
    # 디버깅 정보
    print(f"\n📊 상세 점수:")
    for i, c in enumerate(final_candidates, 1):
        print(f"{i}. {c['name']}: 종합{c['final_score']}/10 (벡터 유사도 {c['vector_similarity']}% + AI 분석 점수 {c['ai_score']}/10)")
    
    if idx < len(data["issues"]) - 1:
        print(f"\n{'-'*80}")

print(f"\n🎉 총 {len(data['issues'])}개 이슈 분석 완료!")
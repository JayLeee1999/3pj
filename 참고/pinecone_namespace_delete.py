import os
from dotenv import load_dotenv
from pinecone import Pinecone

# 환경변수 로드
load_dotenv(override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lastproject")

def delete_namespace(index_name, namespace):
    """특정 namespace의 모든 벡터 삭제"""
    try:
        # Pinecone 클라이언트 초기화
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # 인덱스 연결
        index = pc.Index(index_name)
        
        print(f"🗑️ '{namespace}' namespace 삭제 중...")
        
        # namespace의 모든 벡터 삭제
        index.delete(delete_all=True, namespace=namespace)
        
        print(f"✅ '{namespace}' namespace 삭제 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def delete_multiple_namespaces(index_name, namespaces):
    """여러 namespace를 한번에 삭제"""
    print(f"📋 삭제할 namespace 목록: {namespaces}")
    
    for namespace in namespaces:
        delete_namespace(index_name, namespace)
        print(f"{'='*50}")

def get_index_stats(index_name):
    """인덱스 통계 확인 (삭제 전후 비교용)"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        
        stats = index.describe_index_stats()
        print(f"📊 인덱스 '{index_name}' 통계:")
        print(f"   전체 벡터 수: {stats['total_vector_count']}")
        
        if 'namespaces' in stats:
            print("   Namespace별 벡터 수:")
            for ns, info in stats['namespaces'].items():
                print(f"     - {ns}: {info['vector_count']}개")
        else:
            print("   등록된 namespace가 없습니다.")
            
    except Exception as e:
        print(f"❌ 통계 조회 오류: {e}")

def main():
    print("🚀 Pinecone Namespace 삭제 도구")
    print("="*60)
    
    # 삭제 전 현재 상태 확인
    print("📋 현재 인덱스 상태:")
    get_index_stats(INDEX_NAME)
    print("="*60)
    
    # 삭제할 namespace 목록
    namespaces_to_delete = ["industry", "past_issue"]
    
    # 사용자 확인
    print(f"⚠️  다음 namespace들을 삭제합니다: {namespaces_to_delete}")
    confirm = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        # namespace 삭제 실행
        delete_multiple_namespaces(INDEX_NAME, namespaces_to_delete)
        
        print("="*60)
        print("🔍 삭제 후 인덱스 상태:")
        get_index_stats(INDEX_NAME)
        
    else:
        print("❌ 삭제 작업이 취소되었습니다.")

# 개별 namespace 삭제 함수들
def delete_industry_namespace():
    """industry namespace만 삭제"""
    delete_namespace(INDEX_NAME, "industry")

def delete_past_issue_namespace():
    """past_issue namespace만 삭제"""
    delete_namespace(INDEX_NAME, "past_issue")

if __name__ == "__main__":
    # 메인 실행: 두 namespace 모두 삭제
    main()
    
    # 개별 실행 예시:
    # delete_industry_namespace()    # industry만 삭제
    # delete_past_issue_namespace()  # past_issue만 삭제
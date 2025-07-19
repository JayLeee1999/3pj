from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import traceback
import json
from datetime import datetime
import os

# 1. 크롬 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# 2. 사이트 접속
driver.get("https://www.bigkinds.or.kr/")
time.sleep(2)

# ✅ 강제 스크롤: 오늘의 이슈 섹션이 보이도록 880px 아래로 이동
try:
    driver.execute_script("window.scrollTo(0, 880);")
    time.sleep(1)
    print("✅ 스크롤 이동 완료")
except Exception as e:
    print("❌ 스크롤 이동 실패")
    traceback.print_exc()

# 3. 전체 카테고리 클릭
try:
    category_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.issue-category[data-category="전체"]')))
    driver.execute_script("arguments[0].click();", category_button)
    print("✅ 카테고리 클릭 완료")
    time.sleep(3)
except Exception as e:
    print("❌ 카테고리 클릭 실패")
    traceback.print_exc()

# 4. 이슈 크롤링
results = []
for i in range(1, 11):
    print(f"▶️ {i}번 이슈 처리 시작")
    try:
        # 4-1. 슬라이드 넘기기 (4번부터는 수동으로 넘겨야 보임)
        if i >= 3:
            for _ in range(i - 3):
                try:
                    next_btn = driver.find_element(By.CSS_SELECTOR, 'div.swiper-button-next.section2-btn.st2-sw1-next')
                    is_disabled = next_btn.get_attribute('aria-disabled') == 'true'
                    if is_disabled:
                        break
                    driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(0.8)
                except Exception as e:
                    print(f"⚠️ 슬라이드 넘기기 중 오류 발생 (이슈 {i}): {e}")
                    break

        # 4-2. 이슈 클릭
        issue_selector = f'div.swiper-slide:nth-child({i}) .issue-item-link'
        issue_element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, issue_selector)))
        driver.execute_script("arguments[0].scrollIntoView(true);", issue_element)
        driver.execute_script("arguments[0].click();", issue_element)

        # 4-3. 팝업 내용 및 제목 추출
        title_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'p.issuPopTitle')))
        content_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'p.pT20.issuPopContent')))

        title = title_elem.text.strip()
        content = content_elem.text.strip()

        results.append({
            "이슈번호": i,
            "제목": title,
            "내용": content
        })

        # 4-4. 팝업 닫기
        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        time.sleep(1)

        # 4-5. 팝업 닫은 후 다시 화면을 아래로 스크롤
        driver.execute_script("window.scrollTo(0, 880);")
        time.sleep(1)

    except Exception as e:
        print(f"❌ {i}번 이슈 처리 중 오류 발생:")
        traceback.print_exc()

# 5. 결과 출력
for r in results:
    print(f"\n이슈 {r['이슈번호']} 제목: {r['제목']}")
    print(f"내용:\n{r['내용']}\n{'='*60}")

driver.quit()

# ====== JSON 저장 함수 ======
def save_to_json(data):
    """
    크롤링 결과를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터 (리스트)
    """
    try:
        # 타임스탬프 파일명 생성
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        filename = f"{timestamp}_BigKinds_current_issues.json"
        filepath = os.path.join("data2", filename)
        
        # 메타데이터와 함께 저장
        save_data = {
            "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_issues": len(data),
            "source": "bigkinds.or.kr",
            "category": "전체",
            "issues": data
        }
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON 저장 완료: {filepath}")
        print(f"📊 저장된 이슈 수: {len(data)}개")
        
        return filepath
        
    except Exception as e:
        print(f"❌ JSON 저장 실패: {e}")
        return None

def load_from_json(filepath):
    """
    JSON 파일에서 크롤링 결과 로드
    
    Args:
        filepath: JSON 파일 경로
        
    Returns:
        크롤링 결과 리스트
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON 로드 완료: {filepath}")
        print(f"📊 로드된 이슈 수: {data['total_issues']}개")
        print(f"🕐 크롤링 시간: {data['crawled_at']}")
        
        return data['issues']
        
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}")
        return None

# ====== 사용 예시 ======
if __name__ == "__main__":
    # 크롤링 결과가 results에 저장되어 있다고 가정
    
    # JSON 파일로 저장
    saved_file = save_to_json(results)
    
    # 저장된 파일 로드 테스트
    if saved_file:
        # loaded_data = load_from_json(saved_file)
        # print(f"로드된 첫 번째 이슈: {loaded_data[0] if loaded_data else 'None'}")
        pass
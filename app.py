import streamlit as st 
import pandas as pd
import joblib 
import json

# ----------------------------------------------------
# 1. Gemini 클라이언트 및 타입 임포트 (문제 해결 후 사용)
# ----------------------------------------------------
from google import genai
from google.genai import types 

# Gemini 클라이언트 초기화 (로컬 메모리/CPU 사용 X)
try:
    # secrets.toml 파일에서 API 키를 가져옵니다.
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
    # 로컬 모델 로드가 사라지면서 앱이 훨씬 빠르게 시작됩니다.
except KeyError:
    st.error("❌ 오류: .streamlit/secrets.toml 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    st.info("API 키를 설정한 후 앱을 다시 실행해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gemini 클라이언트 초기화 오류: {e}")
    st.stop()


# ----------------------------------------------------
# 데이터 및 추천 모델 로드 (기존과 동일)
# ----------------------------------------------------
try:
    # CSV 데이터 로드
    clothing_data = pd.read_csv("processed_clothing_data.csv")
    
    # ML 추천 모델 로드
    recommendation_model = joblib.load("clothing_recommendation.pkl")
    st.success("데이터 및 추천 모델 로드 완료!")
    
except Exception as e:
    st.error(f"❌ 데이터 또는 추천 모델 로드 중 오류 발생: {e}")
    clothing_data = pd.DataFrame()
    recommendation_model = None

# ----------------------------------------------------
# 2. Gemini를 사용한 속성 추출 함수 (parse_user_text 대체)
# ----------------------------------------------------
def parse_user_text_gemini(user_text):
    """Gemini API를 사용하여 사용자 문장에서 4가지 속성을 JSON 형태로 추출합니다."""
    
    # JSON 스키마 정의: Gemini에게 원하는 출력 형식을 명확히 알려줍니다.
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "gender": types.Schema(type=types.Type.STRING, description="성별 (male, female)"),
            "season": types.Schema(type=types.Type.STRING, description="계절 (spring, summer, fall, winter)"),
            "style": types.Schema(type=types.Type.STRING, description="스타일 (casual, street, classic, sporty)"),
            "price_range": types.Schema(type=types.Type.STRING, description="가격대 (low, medium, high)"),
        },
        required=["gender", "season", "style", "price_range"]
    )

    prompt = f"다음 의류 추천 문장에서 요청된 4가지 속성을 추출해줘. 문장: '{user_text}'"

    # Gemini API 호출 (구조화된 JSON 출력 요청)
    response = client.models.generate_content(
        model='gemini-2.5-flash', # 빠르고 효율적인 모델 사용
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema
        )
    )
    
    # Gemini는 유효한 JSON 문자열을 반환합니다.
    return json.loads(response.text)

# ----------------------------------------------------
# Streamlit 사용자 인터페이스 (UI) 구성 및 실행
# ----------------------------------------------------

st.set_page_config(layout="wide")
st.title("🤖 Gemini 기반 의류 추천 시스템")

st.markdown("""
### 🗣️ 추천 요청 문구 입력
원하는 의류의 **성별, 계절, 스타일, 가격대**를 문장에 포함하여 입력해주세요.
*예시: 남자 여름용으로 캐주얼하고 10만원대 이하인 옷을 추천해줘.*
""")

# ----------------------------------------------------
# 사용자 입력 및 실행 버튼
# ----------------------------------------------------
user_text = st.text_input("👕 추천 요청 문구:", key="user_input")

if st.button("🚀 속성 추출 및 의류 추천 시작"):
    if not user_text:
        st.warning("문장을 입력해 주세요.")
        st.stop()
    
    # 1. Gemini 속성 추출 단계
    with st.spinner('Gemini API가 텍스트에서 속성을 분석하고 ML 모델 예측을 준비하는 중입니다...'):
        try:
            # Gemini API 호출
            extracted_json = parse_user_text_gemini(user_text)
            
            st.subheader("✅ 1단계: 추출된 속성 확인 (Gemini 결과)")
            st.json(extracted_json)
            
            # 2. ML 추천 단계
            if recommendation_model:
                st.subheader("✨ 2단계: 추천 결과 예측 (ML 모델)")
                
                # LLM 결과 데이터를 DataFrame 형태로 변환
                # 추출된 속성의 순서와 타입이 ML 모델 입력과 일치해야 합니다.
                input_df = pd.DataFrame([extracted_json])
                
                # ML 모델로 예측 실행
                recommendation = recommendation_model.predict(input_df)
                
                st.success(f"**최종 추천 의류 카테고리:** `{recommendation[0]}`")
                
                # 추가 정보 표시
                st.info("추출된 속성과 추천 결과는 화면 아래 데이터 미리보기에서 확인하실 수 있습니다.")
                
            else:
                st.warning("추천 모델이 로드되지 않아 예측을 수행할 수 없습니다.")
                
        except Exception as e:
            st.error(f"❌ 처리 중 알 수 없는 오류 발생 (Gemini API 또는 데이터 처리): {e}")

# ----------------------------------------------------
# 데이터 미리보기 섹션
# ----------------------------------------------------
st.markdown("---")
st.subheader("📚 데이터 미리보기 (`processed_clothing_data.csv`)")

if not clothing_data.empty:
    st.write(f"**총 데이터 수:** {len(clothing_data)}개")
    # 데이터가 어떻게 사용되는지 보여주기 위해 주요 컬럼만 표시
    display_cols = ['gender', 'season', 'style', 'price_range', 'recommended_category']
    st.dataframe(clothing_data[display_cols].head(10)) 
else:
    st.info("데이터 파일 로드에 실패하여 미리보기를 표시할 수 없습니다.")
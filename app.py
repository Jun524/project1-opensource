import streamlit as st 
import pandas as pd
import joblib 
import json
import os
import urllib.parse
from google import genai
from google.genai import types 

# ----------------------------------------------------
# 0. 다중 모델 경로 정의
# ----------------------------------------------------
MODEL_PATHS = {
    "top": "model_top.pkl",
    "bottom": "model_bottom.pkl",
    "outer": "model_outer.pkl"
}

# ----------------------------------------------------
# 1. Gemini 클라이언트 초기화
# ----------------------------------------------------
try:
    # secrets.toml 파일에서 API 키를 가져옵니다.
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except KeyError:
    st.error("❌ 오류: .streamlit/secrets.toml 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    st.info("API 키를 설정한 후 앱을 다시 실행해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gemini 클라이언트 초기화 오류: {e}")
    st.stop()


# ----------------------------------------------------
# 2. 데이터 및 다중 추천 모델 로드 함수 (캐싱 적용)
# ----------------------------------------------------
@st.cache_resource
def load_all_models_and_data():
    """CSV 데이터와 세 가지 PKL 모델을 모두 로드합니다."""
    
    st.info("⏳ 데이터 및 학습된 세 가지 모델을 로드 중입니다...")
    clothing_data = pd.DataFrame()
    recommendation_models = {}
    
    # 2.1 CSV 데이터 로드
    try:
        if os.path.exists("processed_clothing_data.csv"):
            clothing_data = pd.read_csv("processed_clothing_data.csv")
            st.success("✅ CSV 데이터 로드 완료!")
        else:
            st.error("❌ 데이터 오류: 'processed_clothing_data.csv' 파일을 찾을 수 없습니다.")
            return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame(), {}

    # 2.2 다중 ML 모델 로드
    all_loaded = True
    for category, file_name in MODEL_PATHS.items():
        try:
            if not os.path.exists(file_name):
                st.error(f"❌ 모델 로드 오류: **'{file_name}'**을(를) 찾을 수 없습니다.")
                all_loaded = False
                continue
                
            # joblib을 사용하여 모델 로드 (joblib/pickle로 저장된 모델을 로드)
            model = joblib.load(file_name)
            recommendation_models[category] = model
        except Exception as e:
            st.error(f"❌ 모델 로드 중 오류 발생 ({file_name}): {e}")
            all_loaded = False
            
    if all_loaded:
        st.success("✅ 모든 추천 모델 (top, bottom, outer) 로드 완료!")
    
    return clothing_data, recommendation_models

# 데이터 및 모델 로드 실행
CLOTHING_DATA, RECOMMENDATION_MODELS = load_all_models_and_data()

# 모델이 하나도 로드되지 않았을 경우 앱 중단
if not RECOMMENDATION_MODELS:
    st.stop()

# ----------------------------------------------------
# 3. Gemini를 사용한 속성 추출 함수 (color 요청으로 수정)
# ----------------------------------------------------
def parse_user_text_gemini(user_text):
    """Gemini API를 사용하여 사용자 문장에서 4가지 속성을 JSON 형태로 추출합니다. (color 포함)"""
    
    # JSON 스키마 정의: 'season' 대신 'color'를 요청하도록 수정
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "gender": types.Schema(type=types.Type.STRING, description="성별 (male, female)"),
            "color": types.Schema(type=types.Type.STRING, description="색상 (black, white, blue, gray 등)"),
            "style": types.Schema(type=types.Type.STRING, description="스타일 (casual, street, classic, sporty)"),
            "price": types.Schema(type=types.Type.STRING, description="가격대 (low, medium, high 또는 under_50, 50_100 등)"), 
        },
        required=["gender", "color", "style", "price"]
    )

    # 프롬프트도 'color'를 요청하도록 수정
    prompt = f"다음 의류 추천 문장에서 요청된 4가지 속성(gender, color, style, price)을 추출해줘. price는 'under_50', '50_100', '100_200', '200_300', 'over_300' 중 하나로 매핑하고, color는 단일 색상으로 추출해줘. 문장: '{user_text}'"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini API 호출 오류: {e}")
        return None

# ----------------------------------------------------
# 4. 무신사 검색 링크 생성 함수 (가격 필터링 추가)
# ----------------------------------------------------
def generate_musinsa_link(item_type, item_name, gender, style, color, price):
    """
    ML 모델 예측 결과와 Gemini 추출 속성을 조합하여 무신사 검색 링크를 생성합니다.
    가격대 정보를 포함하여 검색 URL을 구성합니다.
    """
    
    # 4.1 가격 범위 매핑 정의 (무신사 쿼리 파라미터에 맞춘 가격대 매핑)
    # 가정: 무신사는 'price' 파라미터에 'min_value~max_value' (원 단위) 형태를 사용한다고 가정합니다.
    PRICE_MAP = {
        'under_50': '0~50000',
        '50_100': '50000~100000',
        '100_200': '100000~200000',
        '200_300': '200000~300000',
        'over_300': '300000~10000000' # 30만원 이상
    }
    
    # 4.2 키워드 검색어 생성
    gender_map = {'male': '남자', 'female': '여자'}
    gender_kr = gender_map.get(gender, '')

    # 키워드 조합: "성별 + 스타일 + 품목 + 색상"
    keywords = [gender_kr, style, item_name, color]
    search_query = " ".join([k for k in keywords if k]).strip() 

    # 4.3 URL 인코딩 및 기본 URL 설정
    encoded_query = urllib.parse.quote(search_query)
    base_url = "https://www.musinsa.com/search/goods"
    
    # 4.4 가격 필터 파라미터 생성
    price_range_param = PRICE_MAP.get(price)
    
    if price_range_param:
        # 검색어(q)와 가격 필터(price)를 모두 포함하여 URL 생성
        # NOTE: 무신사 실제 URL 구조에 따라 'price' 대신 다른 파라미터명(예: filter_price)을 사용해야 할 수 있습니다.
        full_url = f"{base_url}?q={encoded_query}&price={price_range_param}"
    else:
        # 가격 정보가 없거나 매핑되지 않은 경우, 키워드 검색만 수행
        full_url = f"{base_url}?q={encoded_query}"
        
    return full_url, search_query, price_range_param

# ----------------------------------------------------
# Streamlit 사용자 인터페이스 (UI) 구성 및 실행
# ----------------------------------------------------

st.set_page_config(layout="wide")
st.title("🤖 Gemini 기반 의류 추천 시스템")

st.markdown("""
### 🗣️ 추천 요청 문구 입력
원하는 의류의 **성별, 색상, 스타일, 가격대**를 문장에 포함하여 입력해주세요.
*예시: **남자 검은색** 캐주얼한 옷을 **10만원대 이하**로 추천해줘.*
""") 

# ----------------------------------------------------
# 사용자 입력 및 실행 버튼
# ----------------------------------------------------
user_text = st.text_input("👕 추천 요청 문구:", key="user_input")

# 사용자가 어떤 종류의 옷을 추천받고 싶은지 선택 (어떤 ML 모델을 사용할지 결정)
selected_item_type = st.selectbox(
    "어떤 종류의 옷을 추천받고 싶으신가요?",
    list(RECOMMENDATION_MODELS.keys()),
    format_func=lambda x: x.upper(),
    key="item_type_select"
)

if st.button(f"🚀 {selected_item_type.upper()} 의류 추천 시작"):
    if not user_text:
        st.warning("문장을 입력해 주세요.")
        st.stop()
    
    # 1. Gemini 속성 추출 단계
    with st.spinner('Gemini API가 텍스트에서 속성을 분석하고 ML 모델 예측을 준비하는 중입니다...'):
        extracted_json = parse_user_text_gemini(user_text)
        
        if extracted_json is None:
            st.error("속성 추출에 실패했습니다. 입력 문장을 다시 확인해주세요.")
            st.stop()
            
        st.subheader("✅ 1단계: 추출된 속성 확인 (Gemini 결과)")
        st.json(extracted_json)
        
        # ⚠️ ML 모델 입력 데이터 정제 및 변환
        try:
            # ML 모델의 입력 DataFrame 생성: 모델이 요구하는 'gender, style, color, price' 순서를 따름
            input_data = {
                'gender': [extracted_json.get('gender')],
                'style': [extracted_json.get('style')],
                'color': [extracted_json.get('color')],
                'price': [extracted_json.get('price')],
            }
            
            # DataFrame 생성 시 명시적으로 컬럼 순서 지정 (모델 입력 안전성 확보)
            input_df = pd.DataFrame(input_data, columns=['gender', 'style', 'color', 'price'])
            
        except Exception as e:
            st.error(f"❌ ML 모델 입력 데이터 변환 오류: {e}")
            st.stop()
            
        # 2. ML 추천 단계
        current_model = RECOMMENDATION_MODELS.get(selected_item_type)
        final_item = None

        if current_model:
            st.subheader(f"✨ 2단계: 추천 결과 예측 ({selected_item_type.upper()} 모델)")
            
            try:
                # ML 모델로 예측 실행
                recommendation = current_model.predict(input_df)
                
                final_item = recommendation[0]
                
                st.success(f"**최종 추천 의류 품목 ({selected_item_type.upper()}):** `{final_item}`")
                
                # 3. 무신사 링크 생성 단계 (가격 필터링 추가)
                st.subheader("🔗 3단계: 무신사 검색 링크 (가격 필터 포함)")
                
                # Gemini 추출 결과에서 속성 추출
                gender = extracted_json.get('gender', '')
                style = extracted_json.get('style', '')
                color = extracted_json.get('color', '')
                price = extracted_json.get('price', '') # ⬅️ price 속성 추출

                # 링크 생성 함수 호출 시 price 전달
                musinsa_url, search_keywords, price_range = generate_musinsa_link(
                    selected_item_type, 
                    final_item, 
                    gender, 
                    style, 
                    color, 
                    price # ⬅️ price 매개변수 전달
                )
                
                st.markdown(f"**생성된 검색어:** `{search_keywords}`")
                st.markdown(f"**적용된 가격 범위:** `{price_range}` (매핑된 무신사 URL 파라미터)")
                st.markdown(f"**[무신사에서 '{search_keywords}' 검색하기]({musinsa_url})**", unsafe_allow_html=True)
                
                st.info("이제 버튼을 클릭하여 무신사에서 추천된 의류를 바로 확인해 보세요!")
                
            except Exception as e:
                st.error(f"❌ ML 모델 예측 오류: 입력된 속성 값(예: 'gender', 'color' 등의 특정 문자열)이 모델이 학습한 범주에 없습니다. **(에러 상세: {e})**")
                
        else:
            st.warning(f"선택된 카테고리({selected_item_type.upper()})에 해당하는 모델이 로드되지 않았습니다.")
                
# ----------------------------------------------------
# 데이터 미리보기 섹션
# ----------------------------------------------------
st.markdown("---")
st.subheader("📚 데이터 미리보기 (`processed_clothing_data.csv`)")

if not CLOTHING_DATA.empty:
    st.write(f"**총 데이터 수:** {len(CLOTHING_DATA)}개")
    # 모델 학습에 사용되었을 것으로 예상되는 컬럼들을 표시
    display_cols = ['gender', 'style', 'color', 'price', 'top', 'bottom', 'outer']
    
    # 데이터에 실제 존재하는 컬럼만 선택하여 표시
    valid_cols = [col for col in display_cols if col in CLOTHING_DATA.columns]
    
    st.dataframe(CLOTHING_DATA[valid_cols].head(10)) 
else:
    st.info("데이터 파일 로드에 실패하여 미리보기를 표시할 수 없습니다.")
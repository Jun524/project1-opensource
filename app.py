import streamlit as st
import pandas as pd
import joblib
import json
import os
import urllib.parse
from google import genai
from google.genai import types

# ==============================================================================
# 0. UI 테마 설정 (Custom CSS Injection)
# ==============================================================================
custom_css = """
<style>
/* Streamlit 기본 테마를 무시하고 폰트와 배경을 설정 */
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #FFFFFF; /* 배경색: 화이트 */
    color: #000000; /* 기본 텍스트색: 블랙 */
}

/* 메인 제목 스타일 */
h1 {
    color: #000000;
    font-weight: 700;
    border-bottom: 2px solid #000000;
    padding-bottom: 10px;
}

/* 부제목 스타일 */
h2, h3, h4 {
    color: #000000;
    font-weight: 600;
}

/* 사이드바 스타일 (배경색: 아주 미세한 회색) */
[data-testid="stSidebar"] {
    background-color: #F8F8F8;
    border-right: 1px solid #E0E0E0;
}

/* 기본 버튼 스타일: 블랙 배경, 화이트 텍스트 */
[data-testid="baseButton-primary"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #000000 !important;
    border-radius: 4px; /* 약간의 둥글림 */
    transition: all 0.2s;
}
[data-testid="baseButton-primary"]:hover {
    background-color: #333333 !important;
    border-color: #333333 !important;
}

/* 정보/경고/오류 박스 스타일 */
.stAlert, .stNotification {
    border-left: 5px solid #000000; /* 검은색 강조선 */
    background-color: #F0F0F0;
    color: #000000;
}

/* 최종 무신사/네이버 링크 박스 스타일 */
.musinsa-link-box {
    text-align: center;
    padding: 30px;
    background-color: #000000; /* 검은색 배경으로 최종 결과 강조 */
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.musinsa-link-box a {
    display: inline-block;
    padding: 12px 35px;
    background-color: #FFFFFF; /* 흰색 버튼 */
    color: #000000;
    text-decoration: none;
    font-weight: bold;
    border-radius: 4px;
    font-size: 1.3em;
    border: 2px solid #000000;
}
.musinsa-link-box a:hover {
    background-color: #E0E0E0;
}

/* 네이버 쇼핑 링크 박스 (초록색 배경) */
.naver-link-box {
    text-align: center;
    padding: 30px;
    background-color: #03C75A; /* 네이버 초록색 배경 */
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-top: 15px; /* 무신사 박스와 간격 */
}
.naver-link-box a {
    display: inline-block;
    padding: 12px 35px;
    background-color: #FFFFFF; /* 흰색 버튼 */
    color: #03C75A; /* 네이버 색상 텍스트 */
    text-decoration: none;
    font-weight: bold;
    border-radius: 4px;
    font-size: 1.3em;
    border: 2px solid #FFFFFF;
}
.naver-link-box a:hover {
    background-color: #E0E0E0;
}

/* 결과 요약 정보 박스 (st.info) */
[data-testid="stAlert"] {
    border: 1px solid #000000;
    background-color: #FFFFFF;
    color: #000000;
}

/* 메트릭 스타일 (수치 강조) */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    color: #000000 !important;
    font-weight: 700;
}
</style>
"""
# HTML/CSS를 Streamlit 앱에 삽입
st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# 1. 상수 정의 및 Gemini 클라이언트 초기화
# ==============================================================================

# 다중 모델 경로 정의
MODEL_PATHS = {
    "top": "model_top.pkl",
    "bottom": "model_bottom.pkl",
    "outer": "model_outer.pkl"
}


# ✨ 무신사 상위 카테고리 코드 (category 파라미터용)
MUSINSA_CATEGORY_CODES = {
    "top": "001",       # 상의
    "bottom": "002",    # 하의
    "outer": "003"      # 아우터
}

# ✨ 무신사 색상 필터 코드 (color 파라미터용 - 복합 색상 포함)
MUSINSA_COLOR_CODES = {
    "red": "RED,DEEPRED,BRICK,ORANGE", 
    "blue": "BLUE,SKYBLUE,NAVY,DEEPBLUE",
    "black": "BLACK",
    "white": "WHITE,IVORY",
    "gray": "GRAY,CHARCOAL",
    "brown": "BROWN,BEIGE,KHAKI",
    "green": "GREEN,LIGHTGREEN,DEEPGREEN",
    "pink": "PINK,CORAL"
}

# 무신사 필터링을 위한 통합 매핑 정의
MUSINSA_FILTER_MAPPING = {
    "item_kr": {
        "tshirt": "티셔츠", "long_sleeve": "긴팔 티셔츠", "hoodie": "후드 티셔츠", "sweatshirt": "맨투맨", "shirt": "셔츠", "blouse": "블라우스", "crop_top": "크롭탑", "tank_top": "나시", "training_top": "트레이닝 상의",
        "denim": "데님", "slacks": "슬랙스", "cargo_pants": "카고 팬츠", "training_pants": "트레이닝 팬츠", "skirt": "스커트", "shorts": "반바지", "leggings": "레깅스",
        "jacket": "재킷", "padding": "패딩", "blazer": "블레이저", "coat": "코트", "cardigan": "가디건", "zipup_hoodie": "집업 후드", "windbreaker": "바람막이"
    },
    "gender_code": {
        "male": "M", "female": "F"
    },
    "color_kr": {
        "black": "블랙", "white": "화이트", "blue": "블루", "gray": "그레이", "red": "레드", "brown": "브라운", "green": "그린", "pink": "핑크"
    }
}


# Gemini 클라이언트 초기화
try:
    API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except KeyError:
    st.error("❌ 오류: .streamlit/secrets.toml 파일에 'GEMINI_API_KEY'가 설정되어 있지 않습니다.")
    client = None
except Exception as e:
    st.error(f"❌ Gemini 클라이언트 초기화 오류: {e}")
    client = None
    
# ----------------------------------------------------
# 2. 데이터 및 다중 추천 모델 로드 함수 (캐싱 적용)
# ----------------------------------------------------
@st.cache_resource
def load_all_models_and_data():
    clothing_data = pd.DataFrame()
    recommendation_models = {}
    
    # 2.1 CSV 데이터 로드
    try:
        if os.path.exists("processed_clothing_data.csv"):
            clothing_data = pd.read_csv("processed_clothing_data.csv")
        else:
            st.warning("⚠️ 'processed_clothing_data.csv' 파일을 찾을 수 없습니다.")
            pass
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame(), {}

    # 2.2 다중 ML 모델 로드
    for category, file_name in MODEL_PATHS.items():
        try:
            if not os.path.exists(file_name):
                st.error(f"⚠️ **모델 파일 누락:** '{file_name}' 파일을 찾을 수 없습니다. (1단계: train_model.py 실행 필요)")
                continue
                
            model = joblib.load(file_name)
            recommendation_models[category] = model
        except Exception as e:
            st.error(f"❌ '{file_name}' 로드 중 치명적 오류 발생: **{e}**")
            
    return clothing_data, recommendation_models

# 데이터 및 모델 로드 실행
CLOTHING_DATA, RECOMMENDATION_MODELS = load_all_models_and_data()

# ----------------------------------------------------
# 3. Gemini를 사용한 속성 추출 함수
# ----------------------------------------------------
def parse_user_text_gemini(user_text):
    """
    Gemini API를 사용하여 사용자 문장에서 4가지 속성을 JSON 형태로 추출합니다.
    가격은 ML 모델의 입력 카테고리(under_50 등)로 추출합니다.
    """
    
    if not client:
        # Fallback for testing when client is None
        return {"gender": "female", "color": "black", "style": "casual", "price": "under_50"} 

    # JSON 스키마 정의
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "gender": types.Schema(type=types.Type.STRING, description="성별 (male, female)"),
            "color": types.Schema(type=types.Type.STRING, description="색상 (black, white, blue, gray 등)"),
            "style": types.Schema(type=types.Type.STRING, description="스타일 (casual, street, classic, sporty)"),
            "price": types.Schema(type=types.Type.STRING, description="가격대. 가격이 언급되지 않았다면 'under_50'을 기본값으로 반환. 언급되었다면 'under_50', '50_100', '100_200', '200_300', 'over_300' 중 하나로 매핑."),
        },
        required=["gender", "color", "style", "price"]
    )

    # Gemini 프롬프트: 가격 카테고리 명시
    price_keywords = "('under_50', '50_100', '100_200', '200_300', 'over_300' 중 하나. 가격이 언급되지 않았다면 'under_50'을 반환."
    prompt = f"다음 의류 추천 문장에서 요청된 4가지 속성(gender, color, style, price)을 추출해줘. price는 언급되었다면 {price_keywords} 중 하나로 매핑하고, color는 단일 색상으로 추출해줘. (참고: ML 모델은 black, white, blue, gray, red, brown, green, pink만 학습함) 문장: '{user_text}'"

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
        print(f"Gemini API 호출 오류: {e}")
        return None

# ----------------------------------------------------
# 4. 무신사 가격 필터 계산 함수
# ----------------------------------------------------
def get_price_min_max(price_key):
    """
    Gemini price key (ML 모델의 입력 카테고리)를 받아서 무신사 URL에 필요한 price, minPrice, maxPrice 값을 반환합니다.
    """
    MIN_MUSINSA_PRICE = 7200 # 무신사 검색 시 최소 가격

    # ⚠️ ML 학습에 사용된 카테고리에 기반한 가격 범위
    PRICE_BOUNDARIES = {
        'under_50': (0, 50000), 
        '50_100': (50000, 99999), 
        '100_200': (100000, 199999),
        '200_300': (200000, 299999),
        'over_300': (300000, 10000000),
    }

    if not price_key or price_key not in PRICE_BOUNDARIES:
        price_key = 'under_50'
        
    min_b, max_b = PRICE_BOUNDARIES[price_key]
    
    # URL 'price' 파라미터 값 (Min~Max)
    price_param = f"{min_b}~{max_b}"
    
    # minPrice/maxPrice 파라미터 값
    final_min = min_b if min_b > 0 else MIN_MUSINSA_PRICE
    final_max = max_b
    
    return price_param, final_min, final_max

# ----------------------------------------------------
# 5. 무신사 검색 링크 생성 함수
# ----------------------------------------------------
def generate_musinsa_link(item_type, item_name, gender, style, color, price):
    
    # 1. 속성 매핑 및 코드 추출
    item_kr = MUSINSA_FILTER_MAPPING["item_kr"].get(item_name, item_name)
    gender_code = MUSINSA_FILTER_MAPPING["gender_code"].get(gender, "")
    color_kr = MUSINSA_FILTER_MAPPING["color_kr"].get(color, color)

    color_filter_codes = MUSINSA_COLOR_CODES.get(color, "") 
    category_code = MUSINSA_CATEGORY_CODES.get(item_type, "") 
    
    # 가격 정보 추출 (ML 예측 카테고리 기반)
    price_param, min_price, max_price = get_price_min_max(price)

    # 2. URL 파라미터 구성
    base_url = "https://www.musinsa.com/search/goods"
    params = {}
    filter_details = {}
    
    # A. 성별 필터 (gender/gf)
    if gender_code:
        params['gender'] = gender_code
        params['gf'] = gender_code
        filter_details["성별 필터 (gender/gf)"] = gender_code

    # B. 가격 필터 (price, minPrice, maxPrice)
    if price_param:
        params['price'] = price_param
        params['minPrice'] = min_price
        params['maxPrice'] = max_price
        filter_details["가격 필터 (Price Category)"] = price 
        filter_details["가격 범위 (min/max)"] = f"{min_price}~{max_price}"

    # C. 색상 필터 (color)
    if color_filter_codes:
        params['color'] = color_filter_codes 
        filter_details["색상 필터 (color)"] = color_filter_codes
    
    # D. 상위 카테고리 필터 (category)
    if category_code:
        params['category'] = category_code 
        filter_details["상위 카테고리 (category)"] = category_code

    # E. 검색 키워드 (keyword)
    search_keywords = " ".join([k for k in [item_kr, color_kr, style] if k]).strip()
    if search_keywords:
        params['keyword'] = search_keywords 
        filter_details["검색 키워드 (keyword)"] = search_keywords

    # 3. 최종 URL 생성
    encoded_params = urllib.parse.urlencode(params, safe='~')
    full_url = f"{base_url}?{encoded_params}"
        
    return full_url, search_keywords, price_param, filter_details

# ----------------------------------------------------
# 6. ML 모델 예측 함수
# ----------------------------------------------------
def predict_clothing_item(gender, style, color, price, item_type, models):
    """
    로컬 ML 모델을 사용하여 사용자 속성(gender, style, color, price)에 맞는 특정 의류 품목을 예측합니다.
    """
    
    if item_type not in models:
        return "예측 불가 (모델 누락)"

    model = models[item_type]
    
    input_data = {'gender': [gender.lower()], 
                  'style': [style.lower()], 
                  'color': [color.lower()],
                  'price': [price.lower()]} 
    
    input_df = pd.DataFrame(input_data, columns=['gender', 'style', 'color', 'price']) 
    
    try:
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"❌ ML 모델 예측 중 치명적 오류 발생: {e}")
        return "예측 실패"

# ----------------------------------------------------
# 7. Gemini를 사용한 검색 키워드 최적화 함수 (NEW)
# ----------------------------------------------------
def refine_search_query_gemini(product_name):
    """
    Gemini API를 사용하여 사용자가 입력한 제품명을 네이버 쇼핑 검색에 최적화된 형태로 정리합니다.
    """
    if not client:
        return product_name # 클라이언트 없으면 원본 반환

    prompt = f"""
    당신은 쇼핑몰 검색 엔진 최적화 전문가입니다.
    사용자가 입력한 다음 제품명/키워드를 네이버 쇼핑 검색에 가장 적합한 형태로 불필요한 기호나 긴 부연 설명을 제거하고 핵심 키워드만으로 정리해주세요.
    최종 결과는 정리된 텍스트 하나만 반환해야 합니다.

    제품명: '{product_name}'
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        # 줄 바꿈 및 불필요한 공백 제거 후 반환
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        print(f"Gemini 키워드 최적화 오류: {e}")
        return product_name # 오류 발생 시 원본 반환

# ----------------------------------------------------
# 8. 네이버 쇼핑 검색 링크 생성 함수 (최종 수정: 한글 URL 안정화)
# ----------------------------------------------------
def generate_naver_shopping_link(search_keywords):
    """
    최적화된 검색 키워드를 사용하여 네이버 쇼핑 가격 비교 URL을 생성합니다.
    query 파라미터는 urllib.parse.quote_plus로 안정적으로 인코딩됩니다.
    """
    # 네이버 쇼핑 검색 기본 URL
    NAVER_SHOPPING_BASE_URL = "https://search.shopping.naver.com/search/all"
    
    query = search_keywords 
    
    # 💡 핵심 수정: query 파라미터를 수동으로 quote_plus 인코딩하여 한글 링크 오류를 방지
    encoded_query = urllib.parse.quote_plus(query) 
    
    # 네이버 쇼핑 URL 형식: BASE_URL?query=...&cat_id=&frm=NVSHATC
    full_url = f"{NAVER_SHOPPING_BASE_URL}?query={encoded_query}&cat_id=&frm=NVSHATC"
    
    return full_url, query


# ==============================================================================
# 9. Streamlit 사용자 인터페이스 (UI) 구성 및 실행
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🤖 Gemini 기반 무신사 의류 추천기")

# ----------------------------------------------------
# A. 사이드바 (입력 영역)
# ----------------------------------------------------
with st.sidebar:
    st.header("🛍️ 추천 조건 입력")
    st.markdown("""
    원하는 의류의 **성별, 색상, 스타일, 가격대**를 자유롭게 입력해주세요.
    <br><br>
    <span style='font-size: 0.9em;'>
    *예시: **힙한 옷**을 **30만원 이하**로 찾아줘.*
    </span>
    """, unsafe_allow_html=True)
    
    # Session State 초기화 (입력값 유지)
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = "테스트: 여자 캐주얼 버건디 17만원 "
    if 'item_type_select' not in st.session_state:
        st.session_state['item_type_select'] = 'top'

    user_text = st.text_input(
        "📝 추천 요청 문구:",
        key="user_input_widget",
        value=st.session_state['user_input']
    )
    # 입력 값이 변경되면 Session State 업데이트
    if user_text != st.session_state['user_input']:
        st.session_state['user_input'] = user_text
        st.session_state['analysis_done'] = False # 새 입력 시 분석 상태 초기화

    
    # 모델 키가 비어 있는지 확인
    model_keys = list(RECOMMENDATION_MODELS.keys())
    
    # --- 오류 보고 및 SelectBox 설정 ---
    if not model_keys:
        st.error("❌ 오류: 사용 가능한 ML 모델이 없습니다. 모델 파일을 확인해주세요.")
        selected_item_type = None
    else:
        selected_item_type = st.selectbox(
            "🧥 어떤 종류의 옷을 추천받고 싶으신가요?",
            model_keys,
            format_func=lambda x: x.upper(),
            index=model_keys.index(st.session_state['item_type_select']) if st.session_state['item_type_select'] in model_keys else 0,
            key="item_type_select_widget"
        )
    # SelectBox 값이 변경되면 Session State 업데이트
    if selected_item_type != st.session_state['item_type_select']:
        st.session_state['item_type_select'] = selected_item_type
        st.session_state['analysis_done'] = False # 새 선택 시 분석 상태 초기화
    # ------------------------------------------

    st.markdown("---")
    
    button_label = f"🚀 {selected_item_type.upper() if selected_item_type else 'N/A'} 의류 추천 시작"

    run_button = st.button(
        button_label,
        use_container_width=True,
        type="primary",
        disabled=(selected_item_type is None)
    )

# ----------------------------------------------------
# B. 메인 화면 (결과 영역 로직)
# ----------------------------------------------------

# 초기 화면 안내
if not ('analysis_done' in st.session_state and st.session_state['analysis_done']):
    st.markdown("""
    <div style='padding: 20px; border: 1px solid #000000; border-radius: 4px;'>
        <h4 style='margin-top: 0; color: #000000;'>👈 추천을 시작하세요!</h4>
        <p>
        왼쪽 사이드바에 찾고 싶은 의류의 상세 조건을 입력하고 <strong>추천 시작</strong> 버튼을 눌러주세요. <br>
        ✅ 이제 ML 모델은 **4가지 특성(gender, style, color, price)을 모두 사용**하여 예측합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)


# "추천 시작" 버튼이 클릭되었거나, 이미 분석이 완료되어 결과가 Session State에 있는 경우
if run_button or ('analysis_done' in st.session_state and st.session_state['analysis_done']):
    
    # 1. 버튼 클릭 시, 분석 과정 실행 및 세션 상태에 저장
    if run_button:
        # 모든 처리를 st.status 컨테이너 안에서 진행
        with st.status("🔍 추천 엔진 가동 중...", expanded=True) as status:
            
            # 1단계: Gemini 속성 추출
            status.update(label="1/3단계: Gemini 텍스트 속성 분석 중...")
            extracted_json = parse_user_text_gemini(st.session_state['user_input'])
            
            if extracted_json is None:
                status.error("❌ 속성 추출 실패: Gemini API 호출 또는 응답 처리 오류.")
                st.error("속성 추출에 실패했습니다. 입력 문장을 다시 확인하거나 API 키를 점검해주세요.")
                st.session_state['analysis_done'] = False
                st.stop()
                
            gender = extracted_json.get('gender', '')
            style = extracted_json.get('style', '')
            color = extracted_json.get('color', '')
            price = extracted_json.get('price', '') # ML 모델 입력 카테고리
            
            status.update(label="✅ 1/3단계 완료: 속성 추출 성공", state="running")

            # 2단계: ML 추천 예측
            status.update(label="2/3단계: ML 모델로 최종 의류 품목 예측 중...")
            
            try:
                final_item = predict_clothing_item(gender, style, color, price, st.session_state['item_type_select'], RECOMMENDATION_MODELS)
                
                if final_item in ["예측 실패", "예측 불가 (모델 누락)"]:
                    raise ValueError(f"ML 모델 예측 결과가 유효하지 않습니다: {final_item}")
                
                status.update(label="✅ 2/3단계 완료: ML 모델 예측 성공", state="running")
                
            except Exception as e:
                status.error(f"❌ ML 모델 예측 오류: **{e}**")
                st.error(f"ML 모델 예측에 실패했습니다. (상세: {e})")
                st.session_state['analysis_done'] = False
                st.stop()
                
            # 3단계: 무신사 링크 생성
            status.update(label="3/3단계: 무신사 검색 링크 생성 중...")
            
            musinsa_url, search_keywords, price_param, filter_details = generate_musinsa_link(
                st.session_state['item_type_select'], final_item, gender, style, color, price
            )

            # 💡 [핵심 수정] 다음 단계를 위해 모든 분석 결과를 세션 상태에 저장합니다.
            st.session_state['analysis_done'] = True
            st.session_state['final_item'] = final_item
            st.session_state['extracted_json'] = extracted_json
            st.session_state['musinsa_url'] = musinsa_url
            st.session_state['filter_details'] = filter_details
            st.session_state['search_keywords'] = search_keywords # 무신사 검색 키워드

            status.update(label="🎉 모든 단계 완료!", state="complete", expanded=False)

    # 2. Session State에서 데이터 로드 (버튼 클릭/리로드 시 초기화 방지)
    if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
        
        # 세션 상태에서 데이터 로드
        final_item = st.session_state['final_item']
        selected_item_type = st.session_state['item_type_select']
        extracted_json = st.session_state['extracted_json']
        musinsa_url = st.session_state['musinsa_url']
        filter_details = st.session_state['filter_details']
        
        # ----------------------------------------------------
        # C. 최종 결과 요약 및 링크 표시 (무신사)
        # ----------------------------------------------------
        
        st.markdown("## ✨ 최종 추천 결과")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            item_kr_display = MUSINSA_FILTER_MAPPING["item_kr"].get(final_item, final_item)
            st.metric(label=f"추천 의류 품목 ({selected_item_type.upper()})", value=item_kr_display.upper(), delta="ML Prediction (4 Features)")
            
            st.markdown("#### 분석된 조건")
            st.dataframe(
                pd.Series(extracted_json).to_frame().T,
                column_config={
                    "gender": "성별", "color": "색상", "style": "스타일", "price": "가격대(Gemini/ML Key)"
                },
                hide_index=True
            )
            
        with col2:
            st.markdown(f"### 🔗 무신사 바로가기")
            
            st.markdown(f"""
            <div style='padding: 10px; border: 1px solid #E0E0E0; border-radius: 4px; background-color: #F8F8F8;'>
            <p style='margin-bottom: 5px; font-weight: 600;'>✅ **적용된 URL 필터 파라미터:**</p>
            <p style='margin-bottom: 5px; font-weight: 400;'>
                - **상위 카테고리 (category):** {filter_details.get("상위 카테고리 (category)", "N/A")} <br>
                - **성별 (gender/gf):** **{filter_details.get("성별 필터 (gender/gf)", "N/A")}** <br>
                - **가격 범위 (URL Filter):** **{filter_details.get("가격 범위 (min/max)", "N/A")}** <br>
                - **색상 (color):** {filter_details.get("색상 필터 (color)", "N/A")}
            </p>
            <p style='margin-top: 10px; margin-bottom: 5px; font-weight: 600;'>🔍 **검색 키워드 (keyword):**</p>
            <code style='color: #000000; background-color: #FFFFFF; border: 1px solid #000000;'>{filter_details.get("검색 키워드 (keyword)", "N/A")}</code>
            </div>
            """, unsafe_allow_html=True)
            
            # 무신사 링크 버튼
            st.markdown(f"""
            <div class='musinsa-link-box'>
                <a href="{musinsa_url}" target="_blank">
                    무신사에서 추천 결과 확인하기 (새 탭 이동)
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"생성된 URL: `{musinsa_url}`")
            
        # ----------------------------------------------------
        # D. 네이버 쇼핑 가격 비교 (Gemini 키워드 최적화 적용)
        # ----------------------------------------------------
        st.markdown("---")
        st.header("🛒 2단계: 네이버 쇼핑 가격 비교")
        
        item_kr_for_hint = MUSINSA_FILTER_MAPPING["item_kr"].get(final_item, final_item)
        
        st.markdown(f"""
            **☝️ 가격 비교할 의류의 정확한 제품명**을 복사/입력해주세요. <br>
            <span style='color: gray; font-size: 0.9em;'>
            *💡 무신사에서 찾은 제품 이름을 그대로 붙여넣으면 정확합니다. (ML 추천 품목: {item_kr_for_hint.upper()})*
            </span>
            """, unsafe_allow_html=True)
            
        # 💡 [핵심 수정] 사용자 입력창: Session State를 사용하여 값 유지
        if 'naver_shopping_input_value' not in st.session_state:
            st.session_state['naver_shopping_input_value'] = ""

        naver_input = st.text_input(
            "📝 제품명 입력:",
            value=st.session_state['naver_shopping_input_value'], # Session State 값 사용
            key="naver_shopping_input_widget",
            placeholder="여기에 제품명을 붙여넣으세요 (예: '윈드 브레이커 점퍼 & 카고 스커트 세트')"
        )
        
        # 입력 값이 변경되면 세션 상태에 저장 (widget key와 value를 분리 관리)
        if naver_input != st.session_state['naver_shopping_input_value']:
            st.session_state['naver_shopping_input_value'] = naver_input
        
        naver_button = st.button(
            "💚 네이버 쇼핑 가격 비교 시작",
            key="naver_button_widget",
            use_container_width=True
        )

        if naver_button:
            if not st.session_state['naver_shopping_input_value']:
                st.warning("⚠️ 가격 비교할 제품명을 입력해주세요.")
            else:
                # 7. Gemini를 사용하여 검색 키워드 최적화
                with st.spinner('🌟 Gemini가 검색 키워드를 최적화하는 중...'):
                    # 세션 상태에 저장된 입력값 사용
                    optimized_query = refine_search_query_gemini(st.session_state['naver_shopping_input_value'])
                    
                # 8. 네이버 쇼핑 링크 생성 함수 호출 (최적화된 키워드 사용)
                # 💡 [핵심 수정] 수정된 8번 함수 호출
                naver_url, naver_query = generate_naver_shopping_link(optimized_query)
                
                st.markdown(f"**🔍 최적화된 검색 키워드:** `{naver_query}`")
                
                # 네이버 쇼핑 링크 버튼 (초록색 배경)
                # 💡 [핵심 수정] 무신사처럼 새 탭으로 이동하는 HTML 링크 사용
                st.markdown(f"""
                <div class='naver-link-box'>
                    <a href="{naver_url}" target="_blank">
                        네이버 쇼핑에서 가격 비교하기 (새 탭 이동)
                    </a>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"생성된 URL: `{naver_url}`")
            
# ----------------------------------------------------
# E. 데이터 미리보기 섹션 (Expander로 숨김)
# ----------------------------------------------------
st.markdown("---")
with st.expander("📚 데이터 및 모델 정보 미리보기 (개발자용)", expanded=False):
    if not CLOTHING_DATA.empty:
        st.subheader("CSV 데이터 구조")
        st.write(f"**총 데이터 수:** {len(CLOTHING_DATA)}개")
        display_cols = ['gender', 'style', 'color', 'price', 'top', 'bottom', 'outer']
        valid_cols = [col for col in display_cols if col in CLOTHING_DATA.columns]
        st.dataframe(CLOTHING_DATA[valid_cols].head(10))
    else:
        st.info("데이터 파일 로드에 실패했습니다.")

    st.subheader("로드된 ML 모델")
    st.write(f"**총 로드된 모델 수:** {len(RECOMMENDATION_MODELS)}개")
    if RECOMMENDATION_MODELS:
        for name, model in RECOMMENDATION_MODELS.items():
            st.markdown(f"- **{name.upper()} 모델:** 로드 완료 ({type(model).__name__}) (**4 Features 포함**)")
    else:
        st.info("로드된 ML 모델이 없습니다. 위의 오류 메시지를 확인하세요.")
import streamlit as st 
import pandas as pd
import joblib 
import json
import os
import urllib.parse
from google import genai
from google.genai import types 

# ----------------------------------------------------
# 0. UI 테마 설정 (Custom CSS Injection)
# ----------------------------------------------------
# 화이트/블랙 톤앤무드와 깔끔한 스타일을 위한 CSS
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

/* 최종 무신사 링크 박스 스타일 */
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
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except KeyError:
    if os.environ.get("STREAMLIT_SERVER_RUN_ON_SAVE") == "true":
         st.warning("⚠️ API 키가 설정되지 않았습니다. 기능을 사용하려면 'secrets.toml'에 'GEMINI_API_KEY'를 설정해주세요.")
         API_KEY = "PLACEHOLDER_KEY" 
         client = None
    else:
        st.error("❌ 오류: .streamlit/secrets.toml 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
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
    
    clothing_data = pd.DataFrame()
    recommendation_models = {}
    
    # 2.1 CSV 데이터 로드
    try:
        if os.path.exists("processed_clothing_data.csv"):
            clothing_data = pd.read_csv("processed_clothing_data.csv")
        else:
            # CSV 파일 누락 시에는 앱이 실행되지만, 데이터 미리보기에서 오류가 발생할 수 있습니다.
            pass
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame(), {}

    # 2.2 다중 ML 모델 로드 - 파일 존재 여부 및 로드 결과를 명확하게 보고합니다.
    for category, file_name in MODEL_PATHS.items():
        try:
            if not os.path.exists(file_name):
                # 💥 파일이 없을 경우 명시적인 오류 메시지를 사용자에게 표시
                st.error(f"⚠️ **모델 파일 누락:** '{file_name}' 파일을 찾을 수 없습니다. 해당 모델 없이 앱이 실행되거나 추천 기능이 제한될 수 있습니다.")
                continue
                
            model = joblib.load(file_name)
            recommendation_models[category] = model
        except Exception as e:
            st.error(f"❌ '{file_name}' 로드 중 치명적 오류 발생 (손상/형식 오류): **{e}**")
            
    return clothing_data, recommendation_models

# 데이터 및 모델 로드 실행
CLOTHING_DATA, RECOMMENDATION_MODELS = load_all_models_and_data()

# ----------------------------------------------------
# 3. Gemini를 사용한 속성 추출 함수 (color 요청으로 수정)
# ----------------------------------------------------
def parse_user_text_gemini(user_text):
    """Gemini API를 사용하여 사용자 문장에서 4가지 속성을 JSON 형태로 추출합니다. (color 포함)"""
    
    if not client:
         # API 키가 없어 클라이언트가 없을 경우 더미 데이터 반환 (개발 모드)
         return {"gender": "female", "color": "black", "style": "casual", "price": "100_200"}

    # JSON 스키마 정의
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "gender": types.Schema(type=types.Type.STRING, description="성별 (male, female)"),
            "color": types.Schema(type=types.Type.STRING, description="색상 (black, white, blue, gray 등)"),
            "style": types.Schema(type=types.Type.STRING, description="스타일 (casual, street, classic, sporty)"),
            "price": types.Schema(type=types.Type.STRING, description="가격대 (under_50, 50_100, 100_200, 200_300, over_300)"), 
        },
        required=["gender", "color", "style", "price"]
    )

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
        # st.error(f"Gemini API 호출 오류: {e}")
        return None

# ----------------------------------------------------
# 4. 무신사 검색 링크 생성 함수 (가격 필터링 포함)
# ----------------------------------------------------
def generate_musinsa_link(item_type, item_name, gender, style, color, price):
    """
    ML 모델 예측 결과와 Gemini 추출 속성을 조합하여 무신사 검색 링크를 생성하고,
    가격대 정보를 포함하여 검색 URL을 구성합니다.
    """
    
    # 4.1 가격 범위 매핑 정의 (무신사 쿼리 파라미터에 맞춘 가격대 매핑)
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
    # 💥 변수 이름을 search_keywords로 통일
    search_keywords = " ".join([k for k in [gender_kr, style, item_name, color] if k]).strip() 

    # 4.3 URL 인코딩 및 기본 URL 설정
    encoded_query = urllib.parse.quote(search_keywords)
    base_url = "https://www.musinsa.com/search/goods"
    
    # 4.4 가격 필터 파라미터 생성
    price_range_param = PRICE_MAP.get(price)
    
    # URL 구성
    full_url = f"{base_url}?q={encoded_query}"
    
    if price_range_param:
        # 가격 필터 파라미터(price)를 추가합니다.
        full_url += f"&price={price_range_param}"
        
    # 💥 반환 시 search_keywords를 사용
    return full_url, search_keywords, price_range_param

# ----------------------------------------------------
# Streamlit 사용자 인터페이스 (UI) 구성 및 실행
# ----------------------------------------------------

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
    *예시: 남자 검은색 캐주얼한 옷을 10만원대 이하로 찾아줘.*
    </span>
    """, unsafe_allow_html=True)
    
    user_text = st.text_input(
        "📝 추천 요청 문구:", 
        key="user_input",
        value="여자 흰색 클래식한 상의를 5만원대 이하로 추천해줘"
    )
    
    # 모델 키가 비어 있는지 확인
    model_keys = list(RECOMMENDATION_MODELS.keys())
    
    # --- 오류 보고 및 SelectBox 설정 ---
    if not model_keys:
        # 모델이 없을 때 사용자에게 더 명확한 지침을 제공
        st.error("❌ 오류: 사용 가능한 ML 모델이 없습니다. 상의, 하의, 아우터 모델이 모두 누락된 경우입니다. 파일을 업로드했는지 확인해주세요.")
        selected_item_type = None
    else:
        selected_item_type = st.selectbox(
            "🧥 어떤 종류의 옷을 추천받고 싶으신가요?",
            model_keys,
            format_func=lambda x: x.upper(),
            key="item_type_select"
        )
    # ------------------------------------------

    st.markdown("---")
    
    # selected_item_type이 None일 경우를 대비하여 안전하게 .upper() 호출
    button_label = f"🚀 {selected_item_type.upper() if selected_item_type else 'N/A'} 의류 추천 시작"

    # 모델이 로드되지 않았으면 버튼을 비활성화합니다.
    run_button = st.button(
        button_label, 
        use_container_width=True, 
        type="primary", 
        disabled=(selected_item_type is None) # selected_item_type이 None이면 버튼 비활성화
    )

# ----------------------------------------------------
# B. 메인 화면 (결과 영역)
# ----------------------------------------------------

# 초기 화면 안내
if not run_button:
    st.markdown("""
    <div style='padding: 20px; border: 1px solid #000000; border-radius: 4px;'>
        <h4 style='margin-top: 0; color: #000000;'>👈 추천을 시작하세요!</h4>
        <p>
        왼쪽 사이드바에 찾고 싶은 의류의 상세 조건을 입력하고 <strong>추천 시작</strong> 버튼을 눌러주세요. <br>
        Gemini AI와 ML 모델의 2단계 분석을 통해 사용자 맞춤형 의류를 추천해드립니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif selected_item_type is None:
    # 모델이 없는데 버튼이 클릭된 경우 (비활성화 상태라 클릭 불가능해야 함)
    st.error("❌ 추천을 진행할 수 없습니다. 로드된 ML 모델이 없어 버튼이 비활성화되었습니다.")
    st.stop()
else:
    if not user_text:
        st.warning("문장을 입력해 주세요.")
        st.stop()

    # 모든 처리를 st.status 컨테이너 안에서 진행하여 사용자에게 진행 상황을 명확하게 보여줍니다.
    with st.status("🔍 추천 엔진 가동 중...", expanded=True) as status:
        
        # ----------------------------------------------------
        # 1단계: Gemini 속성 추출
        # ----------------------------------------------------
        status.update(label="1/3단계: Gemini 텍스트 속성 분석 중...")
        extracted_json = parse_user_text_gemini(user_text)
        
        if extracted_json is None:
            status.error("❌ 속성 추출 실패: Gemini API 호출 또는 응답 처리 오류.")
            st.error("속성 추출에 실패했습니다. 입력 문장을 다시 확인하거나 API 키를 점검해주세요.")
            st.stop()
            
        status.update(label="✅ 1/3단계 완료: 속성 추출 성공")
        
        # ----------------------------------------------------
        # 2단계: ML 추천 예측
        # ----------------------------------------------------
        status.update(label="2/3단계: ML 모델로 최종 의류 품목 예측 중...")
        
        try:
            # ML 모델의 입력 DataFrame 생성
            input_data = {
                'gender': [extracted_json.get('gender')],
                'style': [extracted_json.get('style')],
                'color': [extracted_json.get('color')],
                'price': [extracted_json.get('price')],
            }
            input_df = pd.DataFrame(input_data, columns=['gender', 'style', 'color', 'price'])
            
            current_model = RECOMMENDATION_MODELS.get(selected_item_type)
            if not current_model:
                # 이 경우는 선택된 모델이 실제로 로드되지 않았을 때 발생 (load_all_models_and_data에서 에러 보고됨)
                raise ValueError(f"모델 '{selected_item_type}'이(가) 로드되지 않았습니다. 추천을 진행할 수 없습니다.")

            # ML 모델로 예측 실행
            recommendation = current_model.predict(input_df)
            final_item = recommendation[0]
            
            status.update(label="✅ 2/3단계 완료: ML 모델 예측 성공", state="running")
            
        except Exception as e:
            status.error(f"❌ ML 모델 예측 오류: **{e}**")
            st.error(f"ML 모델 예측에 실패했습니다. 입력된 속성 값이 모델 학습 범위에 없는 값일 수 있습니다. (상세: {e})")
            st.stop()
            
        # ----------------------------------------------------
        # 3단계: 무신사 링크 생성
        # ----------------------------------------------------
        status.update(label="3/3단계: 무신사 검색 링크 생성 중...")
        
        gender = extracted_json.get('gender', '')
        style = extracted_json.get('style', '')
        color = extracted_json.get('color', '')
        price = extracted_json.get('price', '') 

        # 링크 생성 함수 호출 (반환 변수 순서 및 이름 통일)
        musinsa_url, search_keywords, price_range = generate_musinsa_link(
            selected_item_type, 
            final_item, 
            gender, 
            style, 
            color, 
            price
        )

        status.update(label="🎉 모든 단계 완료!", state="complete", expanded=False)


    # ----------------------------------------------------
    # C. 최종 결과 요약 및 링크 표시
    # ----------------------------------------------------
    
    st.markdown("## ✨ 최종 추천 결과")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label=f"추천 의류 품목 ({selected_item_type.upper()})", value=final_item.upper(), delta="ML Prediction")
        
        st.markdown("#### 분석된 조건")
        st.dataframe(
            pd.Series(extracted_json).to_frame().T, 
            column_config={
                "gender": "성별", "color": "색상", "style": "스타일", "price": "가격대"
            },
            hide_index=True
        )
        
    with col2:
        st.markdown(f"### 🔗 무신사 바로가기")
        
        st.markdown(f"""
        <div style='padding: 10px; border: 1px solid #E0E0E0; border-radius: 4px; background-color: #F8F8F8;'>
        <p style='margin-bottom: 5px; font-weight: 600;'>🔍 검색 키워드:</p> 
        <code style='color: #000000; background-color: #FFFFFF; border: 1px solid #000000;'>{search_keywords}</code>
        <p style='margin-top: 10px; margin-bottom: 5px; font-weight: 600;'>💰 가격 필터 (파라미터):</p>
        <code style='color: #000000; background-color: #FFFFFF; border: 1px solid #000000;'>{price_range}</code> ({price} 매핑)
        </div>
        """, unsafe_allow_html=True)
        
        # 링크 버튼 (주요 행동 유도) - 커스텀 CSS 적용
        st.markdown(f"""
        <div class='musinsa-link-box'>
            <a href="{musinsa_url}" target="_blank">
                무신사에서 추천 결과 확인하기
            </a>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# D. 데이터 미리보기 섹션 (Expander로 숨김)
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
            st.markdown(f"- **{name.upper()} 모델:** 로드 완료 ({type(model).__name__})")
    else:
        st.info("로드된 ML 모델이 없습니다. 위의 오류 메시지를 확인하세요.")
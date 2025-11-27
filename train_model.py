import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
try:
    # 파일 이름을 'processed_clothing_data.csv'로 가정합니다.
    df = pd.read_csv("processed_clothing_data.csv")
    print(f"데이터 로드 완료. 총 {len(df)}개 데이터.")
except FileNotFoundError:
    print("❌ 오류: 'processed_clothing_data.csv' 파일을 찾을 수 없습니다.")
    exit()

# Common input features (X) - ⚠️ PRICE 다시 포함
X = df[["gender", "style", "color", "price"]]
print(f"입력 특성: {list(X.columns)} (총 {len(X.columns)}개)")

# Targets (y)
y_top = df["top"]
y_bottom = df["bottom"]
y_outer = df["outer"]

# ---------------------------------------------------
# 2. Preprocessing (One-Hot Encoding) - ⚠️ PRICE 다시 포함
# ---------------------------------------------------
# 4가지 컬럼 모두 전처리 대상입니다.
categorical_cols = ["gender", "style", "color", "price"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)
print("전처리 파이프라인 정의 완료: OneHotEncoder (gender, style, color, price)")


# ---------------------------------------------------
# 3. Create Model Builder Function
# ---------------------------------------------------
def build_and_train_model(X, y):
    """모델 파이프라인(전처리 + 분류기)을 생성하고 학습합니다."""
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X, y)
    return model

# ---------------------------------------------------
# 4. Train Models
# ---------------------------------------------------
print("\n--- 모델 학습 시작 ---")
print("Top 모델 학습 중...")
model_top = build_and_train_model(X, y_top)

print("Bottom 모델 학습 중...")
model_bottom = build_and_train_model(X, y_bottom)

print("Outer 모델 학습 중...")
model_outer = build_and_train_model(X, y_outer)
print("--- 모델 학습 완료 ---")

# ---------------------------------------------------
# 5. Save Models
# ---------------------------------------------------
joblib.dump(model_top, "model_top.pkl")
joblib.dump(model_bottom, "model_bottom.pkl")
joblib.dump(model_outer, "model_outer.pkl")

print("\n✅ 모든 모델이 'model_top.pkl', 'model_bottom.pkl', 'model_outer.pkl'로 저장되었습니다.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1) CSV 불러오기
df = pd.read_csv("processed_clothing_data.csv")

# 2) X(입력), y(정답) 나누기
X = df.drop("recommended_category", axis=1)
y = df["recommended_category"]

# 3) 범주형 컬럼
cat_features = ["gender", "season", "style", "price_range"]

# 4) One-hot 인코딩 + 랜덤포레스트 파이프라인
model = Pipeline([
    ("encoder", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="drop"
    )),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# 5) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 6) 모델 학습
model.fit(X_train, y_train)

# 7) 모델 정확도 출력
print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))

# 8) 모델 저장
joblib.dump(model, "clothing_recommendation.pkl")
print("모델 저장 완료 → clothing_recommendation.pkl")

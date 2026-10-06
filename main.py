import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 이용하여 과거 기간별 선형회귀 모델을 만들고 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

# --------------------------------------------------
# 연도별 평균기온
# 관측일수가 300일 미만인 연도는 제외
# 2025년 이후 데이터도 제외
# --------------------------------------------------

yearly = (
    df.dropna(subset=["연도", "평균기온"])
      .groupby("연도")
      .agg(
          연평균기온=("평균기온", "mean"),
          관측일수=("평균기온", "count")
      )
      .reset_index()
)

yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly["연도"] = yearly["연도"].astype(int)

yearly = yearly.sort_values("연도").reset_index(drop=True)

# --------------------------------------------------
# 전체 데이터 현황
# --------------------------------------------------

st.subheader("📊 전체 연평균기온 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "사용 가능한 연도 수",
        f"{len(yearly)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{yearly['연도'].min()}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{yearly['연도'].max()}년"
    )

# --------------------------------------------------
# 학습 / 테스트 데이터 분리
#
# 50년 모델: 1956~2005
# 100년 모델: 1906~2005
# 테스트: 2006~2025
# --------------------------------------------------

train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()

# 데이터가 부족한 경우
if len(train_50) < 2 or len(train_100) < 2 or len(test) < 2:
    st.error("회귀분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()

# --------------------------------------------------
# 독립변수
# 연도를 그대로 사용
# --------------------------------------------------

X_50 = train_50[["연도"]]
y_50 = train_50["연평균기온"]

X_100 = train_100[["연도"]]
y_100 = train_100["연평균기온"]

X_test = test[["연도"]]
y_test = test["연평균기온"]

# --------------------------------------------------
# 선형회귀 모델
# --------------------------------------------------

model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(X_50, y_50)
model_100.fit(X_100, y_100)

# 테스트 데이터 예측
pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)

# --------------------------------------------------
# 평가 지표
# --------------------------------------------------

mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)

# --------------------------------------------------
# 회귀선 정보
# --------------------------------------------------

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_

# --------------------------------------------------
# 학습 데이터와 테스트 데이터 설명
# --------------------------------------------------

st.subheader("📚 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}년"
    )
    st.caption("1956~2005")

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}년"
    )
    st.caption("1906~2005")

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}년"
    )
    st.caption("2006~2025")

st.info(
    "두 모델은 모두 2005년까지의 데이터만 학습하고, "
    "동일한 2006~2025년 데이터를 테스트 데이터로 사용합니다."
)

# --------------------------------------------------
# 평가 결과
# --------------------------------------------------

st.subheader("📈 테스트 데이터 예측 성능")

st.write("테스트 데이터: **2006~2025년 연평균기온**")

st.markdown("### 최근 50년 학습 모델")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae_50:.3f} °C"
    )

with col2:
    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )

with col3:
    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )

st.markdown("### 최근 100년 학습 모델")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae_100:.3f} °C"
    )

with col2:
    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )

with col3:
    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )

# --------------------------------------------------
# 회귀선 기울기 비교
# --------------------------------------------------

st.subheader("📐 회귀선 기울기 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "학습 연도 수": [
        len(train_50),
        len(train_100)
    ],
    "기울기 (°C/년)": [
        slope_50,
        slope_100
    ],
    "절편": [
        intercept_50,
        intercept_100
    ],
    "MAE": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "기울기 (°C/년)": "{:.5f}",
        "절편": "{:.3f}",
        "MAE": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 기울기 해석
# --------------------------------------------------

slope_difference = slope_50 - slope_100

if slope_difference > 0:
    slope_text = (
        f"최근 50년 모델의 기울기가 100년 모델보다 "
        f"{slope_difference:.5f} °C/년 더 큽니다."
    )
elif slope_difference < 0:
    slope_text = (
        f"최근 100년 모델의 기울기가 50년 모델보다 "
        f"{abs(slope_difference):.5f} °C/년 더 큽니다."
    )
else:
    slope_text = "두 모델의 기울기는 동일합니다."

st.info(slope_text)

# --------------------------------------------------
# 실제값 vs 예측값 그래프
# --------------------------------------------------

st.subheader("📉 2006~2025년 실제 기온과 예측값 비교")

fig_test = go.Figure()

# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            "%{x}년<br>"
            "실제 평균기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="50년 학습 모델",
        line=dict(dash="dash"),
        hovertemplate=(
            "%{x}년<br>"
            "50년 모델 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 100년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="100년 학습 모델",
        line=dict(dash="dot"),
        hovertemplate=(
            "%{x}년<br>"
            "100년 모델 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig_test.update_layout(
    title="공통 테스트 데이터(2006~2025) 예측 결과",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    height=600,
    hovermode="x unified"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)

# --------------------------------------------------
# 회귀선 자체 비교
# --------------------------------------------------

st.subheader("📈 두 학습기간의 회귀선 비교")

plot_years = np.arange(
    min(train_100["연도"].min(), train_50["연도"].min()),
    2026
)

pred_line_50 = model_50.predict(
    pd.DataFrame({"연도": plot_years})
)

pred_line_100 = model_100.predict(
    pd.DataFrame({"연도": plot_years})
)

fig_reg = go.Figure()

# 50년 회귀선
fig_reg.add_trace(
    go.Scatter(
        x=plot_years,
        y=pred_line_50,
        mode="lines",
        name="1956~2005 학습",
        line=dict(width=3)
    )
)

# 100년 회귀선
fig_reg.add_trace(
    go.Scatter(
        x=plot_years,
        y=pred_line_100,
        mode="lines",
        name="1906~2005 학습",
        line=dict(width=3, dash="dash")
    )
)

# 실제 데이터
fig_reg.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=5, opacity=0.5)
    )
)

fig_reg.update_layout(
    title="50년 학습 회귀선 vs 100년 학습 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=650,
    hovermode="x unified"
)

st.plotly_chart(
    fig_reg,
    use_container_width=True
)

# --------------------------------------------------
# 최종 해석
# --------------------------------------------------

st.subheader("🔎 결과 해석")

# MAE 비교
if mae_50 < mae_100:
    mae_result = "50년 모델의 MAE가 더 작아 평균적인 예측 오차가 더 작았습니다."
elif mae_50 > mae_100:
    mae_result = "100년 모델의 MAE가 더 작아 평균적인 예측 오차가 더 작았습니다."
else:
    mae_result = "두 모델의 MAE가 동일했습니다."

# MSE 비교
if mse_50 < mse_100:
    mse_result = "50년 모델의 MSE가 더 작아 큰 예측 오차도 상대적으로 적었습니다."
elif mse_50 > mse_100:
    mse_result = "100년 모델의 MSE가 더 작아 큰 예측 오차가 상대적으로 적었습니다."
else:
    mse_result = "두 모델의 MSE가 동일했습니다."

# R2 비교
if r2_50 > r2_100:
    r2_result = "50년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명했습니다."
elif r2_50 < r2_100:
    r2_result = "100년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명했습니다."
else:
    r2_result = "두 모델의 R²가 동일했습니다."

st.write(f"**MAE:** {mae_result}")
st.write(f"**MSE:** {mse_result}")
st.write(f"**R²:** {r2_result}")

st.write(
    f"**기울기:** 50년 모델은 연평균 약 "
    f"{slope_50:.4f}°C/년, "
    f"100년 모델은 연평균 약 "
    f"{slope_100:.4f}°C/년의 변화 추세를 나타냅니다."
)

st.caption(
    "※ 2025년까지의 데이터 중 연간 관측일수가 300일 이상인 연도만 사용했습니다. "
    "테스트 데이터는 두 모델에서 동일하게 2006~2025년을 사용했습니다."
)

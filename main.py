import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------
# 페이지 설정
# ----------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 이용하여 50년 학습 모델과 "
    "100년 학습 모델의 예측 성능을 비교합니다."
)

# ----------------------------------------
# 데이터 불러오기
# ----------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception as e:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.stop()


# ----------------------------------------
# 연도별 평균기온 계산
# 관측일수가 300일 이상인 연도만 사용
# 2025년 이후는 제외
# ----------------------------------------

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


# ----------------------------------------
# 학습 / 테스트 데이터
# ----------------------------------------

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


# 데이터 확인
if len(train_50) < 2:
    st.error("1956~2005년의 학습 데이터가 부족합니다.")
    st.stop()

if len(train_100) < 2:
    st.error("1906~2005년의 학습 데이터가 부족합니다.")
    st.stop()

if len(test) < 2:
    st.error("2006~2025년의 테스트 데이터가 부족합니다.")
    st.stop()


# ----------------------------------------
# 선형회귀 함수
# y = ax + b
# ----------------------------------------

def make_regression(data):

    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    # np.polyfit으로 1차 선형회귀
    slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


# 50년 모델
slope_50, intercept_50 = make_regression(train_50)

# 100년 모델
slope_100, intercept_100 = make_regression(train_100)


# ----------------------------------------
# 예측 함수
# ----------------------------------------

def predict(years, slope, intercept):

    years = np.asarray(years, dtype=float)

    return slope * years + intercept


# 테스트 데이터 예측
test_years = test["연도"].to_numpy()

actual = test["연평균기온"].to_numpy()

pred_50 = predict(
    test_years,
    slope_50,
    intercept_50
)

pred_100 = predict(
    test_years,
    slope_100,
    intercept_100
)


# ----------------------------------------
# 평가 지표 함수
# ----------------------------------------

def calculate_metrics(actual, predicted):

    # MAE
    mae = np.mean(
        np.abs(actual - predicted)
    )

    # MSE
    mse = np.mean(
        (actual - predicted) ** 2
    )

    # R²
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# 50년 모델 평가
mae_50, mse_50, r2_50 = calculate_metrics(
    actual,
    pred_50
)

# 100년 모델 평가
mae_100, mse_100, r2_100 = calculate_metrics(
    actual,
    pred_100
)


# ========================================
# 화면
# ========================================

st.subheader("📚 데이터 분할")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}개 연도"
    )
    st.caption("1956~2005")

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}개 연도"
    )
    st.caption("1906~2005")

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )
    st.caption("2006~2025")


st.info(
    "두 모델 모두 2005년까지의 데이터만 학습하고, "
    "동일한 2006~2025년 데이터를 이용하여 평가합니다."
)


# ========================================
# 회귀식과 기울기
# ========================================

st.subheader("📐 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 최근 50년 모델")

    st.metric(
        "기울기",
        f"{slope_50:.5f} °C/년"
    )

    st.write(
        f"연평균기온 = "
        f"{slope_50:.5f} × 연도 "
        f"+ {intercept_50:.3f}"
    )

with col2:

    st.markdown("### 최근 100년 모델")

    st.metric(
        "기울기",
        f"{slope_100:.5f} °C/년"
    )

    st.write(
        f"연평균기온 = "
        f"{slope_100:.5f} × 연도 "
        f"+ {intercept_100:.3f}"
    )


# ========================================
# 성능 평가
# ========================================

st.subheader("🎯 테스트 데이터 예측 성능")

st.write("테스트 데이터: **2006~2025년**")

# 50년
st.markdown("### 🟦 최근 50년 학습 모델")

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


# 100년
st.markdown("### 🟩 최근 100년 학습 모델")

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


# ========================================
# 비교표
# ========================================

st.subheader("📊 두 모델 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 (1956~2005)",
        "최근 100년 (1906~2005)"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기 (°C/년)": [
        slope_50,
        slope_100
    ],
    "MAE (°C)": [
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
        "MAE (°C)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ========================================
# 실제값 vs 예측값
# ========================================

st.subheader("📈 2006~2025년 실제 기온과 예측값")

fig = go.Figure()

# 실제값
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=actual,
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(width=3),
        marker=dict(size=7)
    )
)

# 50년 모델
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="50년 학습 모델",
        line=dict(
            width=3,
            dash="dash"
        )
    )
)

# 100년 모델
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="100년 학습 모델",
        line=dict(
            width=3,
            dash="dot"
        )
    )
)

fig.update_layout(
    title="공통 테스트 데이터의 실제값과 예측값",
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
    fig,
    use_container_width=True
)


# ========================================
# 회귀선 비교
# ========================================

st.subheader("📉 50년 학습 vs 100년 학습 회귀선")

plot_years = np.arange(
    1906,
    2026
)

line_50 = predict(
    plot_years,
    slope_50,
    intercept_50
)

line_100 = predict(
    plot_years,
    slope_100,
    intercept_100
)

fig2 = go.Figure()

# 실제 연평균기온
fig2.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=5
        )
    )
)

# 50년 회귀선
fig2.add_trace(
    go.Scatter(
        x=plot_years,
        y=line_50,
        mode="lines",
        name="1956~2005 회귀선",
        line=dict(
            width=3,
            dash="dash"
        )
    )
)

# 100년 회귀선
fig2.add_trace(
    go.Scatter(
        x=plot_years,
        y=line_100,
        mode="lines",
        name="1906~2005 회귀선",
        line=dict(
            width=3
        )
    )
)

fig2.update_layout(
    title="학습기간에 따른 회귀선 비교",
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
    fig2,
    use_container_width=True
)


# ========================================
# 결과 해석
# ========================================

st.subheader("🔎 결과 해석")

# MAE
if mae_50 < mae_100:
    mae_text = (
        "최근 50년 모델의 MAE가 더 작아 "
        "2006~2025년의 실제 기온을 평균적으로 더 정확하게 예측했습니다."
    )
elif mae_100 < mae_50:
    mae_text = (
        "최근 100년 모델의 MAE가 더 작아 "
        "2006~2025년의 실제 기온을 평균적으로 더 정확하게 예측했습니다."
    )
else:
    mae_text = "두 모델의 MAE가 같습니다."


# MSE
if mse_50 < mse_100:
    mse_text = (
        "최근 50년 모델의 MSE가 더 작아 "
        "큰 예측 오차가 상대적으로 적었습니다."
    )
elif mse_100 < mse_50:
    mse_text = (
        "최근 100년 모델의 MSE가 더 작아 "
        "큰 예측 오차가 상대적으로 적었습니다."
    )
else:
    mse_text = "두 모델의 MSE가 같습니다."


# R²
if r2_50 > r2_100:
    r2_text = (
        "최근 50년 모델의 R²가 더 높아 "
        "테스트 기간의 기온 변화를 더 잘 설명했습니다."
    )
elif r2_100 > r2_50:
    r2_text = (
        "최근 100년 모델의 R²가 더 높아 "
        "테스트 기간의 기온 변화를 더 잘 설명했습니다."
    )
else:
    r2_text = "두 모델의 R²가 같습니다."


st.write("**MAE:** " + mae_text)
st.write("**MSE:** " + mse_text)
st.write("**R²:** " + r2_text)

slope_difference = slope_50 - slope_100

st.write(
    f"**기울기 차이:** "
    f"{slope_difference:.5f} °C/년"
)

if slope_50 > slope_100:
    st.write(
        "최근 50년을 학습한 회귀선이 최근 100년을 학습한 회귀선보다 "
        "더 가파른 상승 추세를 나타냅니다."
    )
elif slope_50 < slope_100:
    st.write(
        "최근 100년을 학습한 회귀선이 최근 50년을 학습한 회귀선보다 "
        "더 가파른 상승 추세를 나타냅니다."
    )
else:
    st.write(
        "두 회귀선의 기울기는 같습니다."
    )


st.caption(
    "※ 2025년까지의 데이터 중 연평균기온을 계산할 때 "
    "관측일수가 300일 이상인 연도만 사용했습니다."
)

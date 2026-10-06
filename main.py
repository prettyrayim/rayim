import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기 - 다항회귀",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기 - 직선 vs 곡선")

st.write(
    "서울의 연평균기온을 이용하여 1차, 3차, 9차 다항회귀 모델을 만들고 "
    "학습에 사용하지 않은 테스트 데이터로 예측 성능을 비교합니다."
)

# --------------------------------------------------
# 데이터 주소
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.stop()


# --------------------------------------------------
# 연평균기온 계산
# 조건
# ① 2025년 이후 제외
# ② 관측일수가 300일 미만인 연도 제외
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
# 훈련 / 테스트 데이터 분리
#
# 2005년 이전 = 훈련
# 2005년부터 = 테스트
# --------------------------------------------------
train = yearly[
    yearly["연도"] < 2005
].copy()

test = yearly[
    yearly["연도"] >= 2005
].copy()


# --------------------------------------------------
# 데이터 개수 확인
# --------------------------------------------------
st.subheader("📚 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )
    st.caption(
        f"{train['연도'].min()}~{train['연도'].max()}년"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )
    st.caption(
        f"{test['연도'].min()}~{test['연도'].max()}년"
    )

st.info(
    "2005년까지가 아니라 **2004년까지** 모델을 학습하고, "
    "2005~2025년은 학습에 전혀 사용하지 않은 테스트 데이터로 평가합니다."
)


# --------------------------------------------------
# 수치 안정화를 위한 연도 변환
#
# 실제 연도:
# 1908, 1909, ..., 2025
#
# 계산용:
# (연도 - 2000) / 100
#
# 예:
# 1900 -> -1
# 2000 -> 0
# 2050 -> 0.5
# --------------------------------------------------
def scale_year(year):
    return (np.asarray(year, dtype=float) - 2000) / 100


# --------------------------------------------------
# 다항회귀 함수
# --------------------------------------------------
def make_polynomial_model(data, degree):

    x = scale_year(data["연도"])
    y = data["연평균기온"].to_numpy(dtype=float)

    coefficients = np.polyfit(
        x,
        y,
        degree
    )

    return coefficients


# --------------------------------------------------
# 예측 함수
# --------------------------------------------------
def predict(coefficients, years):

    x = scale_year(years)

    return np.polyval(
        coefficients,
        x
    )


# --------------------------------------------------
# 1차 / 3차 / 9차 모델 학습
# --------------------------------------------------
coef_1 = make_polynomial_model(
    train,
    1
)

coef_3 = make_polynomial_model(
    train,
    3
)

coef_9 = make_polynomial_model(
    train,
    9
)


# --------------------------------------------------
# 테스트 데이터 예측
# --------------------------------------------------
test_years = test["연도"].to_numpy()
actual = test["연평균기온"].to_numpy()

pred_1 = predict(
    coef_1,
    test_years
)

pred_3 = predict(
    coef_3,
    test_years
)

pred_9 = predict(
    coef_9,
    test_years
)


# --------------------------------------------------
# MAE 계산
# --------------------------------------------------
def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(actual - predicted)
    )


mae_1 = calculate_mae(
    actual,
    pred_1
)

mae_3 = calculate_mae(
    actual,
    pred_3
)

mae_9 = calculate_mae(
    actual,
    pred_9
)


# --------------------------------------------------
# 2050년 예측
# --------------------------------------------------
prediction_2050_1 = predict(
    coef_1,
    [2050]
)[0]

prediction_2050_3 = predict(
    coef_3,
    [2050]
)[0]

prediction_2050_9 = predict(
    coef_9,
    [2050]
)[0]


# --------------------------------------------------
# 결과 표
# --------------------------------------------------
st.subheader("🎯 모델별 테스트 성능과 2050년 예측")

result = pd.DataFrame({
    "모델": [
        "1차 (직선)",
        "3차 곡선",
        "9차 곡선"
    ],
    "훈련 연도 수": [
        len(train),
        len(train),
        len(train)
    ],
    "테스트 연도 수": [
        len(test),
        len(test),
        len(test)
    ],
    "테스트 MAE (°C)": [
        mae_1,
        mae_3,
        mae_9
    ],
    "2050년 예측기온 (°C)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})

st.dataframe(
    result.style.format({
        "테스트 MAE (°C)": "{:.3f}",
        "2050년 예측기온 (°C)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 가장 좋은 모델
# --------------------------------------------------
mae_values = {
    "1차": mae_1,
    "3차": mae_3,
    "9차": mae_9
}

best_model = min(
    mae_values,
    key=mae_values.get
)

best_mae = mae_values[best_model]

st.success(
    f"테스트 데이터의 MAE가 가장 작은 모델은 "
    f"**{best_model} 회귀**이며, MAE는 **{best_mae:.3f}°C**입니다."
)


# --------------------------------------------------
# 실제값 + 세 가지 회귀선
# --------------------------------------------------
st.subheader("📈 실제 연평균기온과 세 가지 회귀선")

# 1900~2050 범위에서 회귀선 표시
plot_years = np.arange(
    1900,
    2051
)

plot_1 = predict(
    coef_1,
    plot_years
)

plot_3 = predict(
    coef_3,
    plot_years
)

plot_9 = predict(
    coef_9,
    plot_years
)


fig = go.Figure()


# 실제값
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=5
        ),
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 1차
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_1,
        mode="lines",
        name="1차 회귀",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "%{x}년<br>"
            "1차 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 3차
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_3,
        mode="lines",
        name="3차 회귀",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "3차 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 9차
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_9,
        mode="lines",
        name="9차 회귀",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "9차 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 2050년 표시
fig.add_trace(
    go.Scatter(
        x=[2050],
        y=[prediction_2050_1],
        mode="markers",
        name="2050년 1차 예측",
        marker=dict(
            size=12,
            symbol="star"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=[2050],
        y=[prediction_2050_3],
        mode="markers",
        name="2050년 3차 예측",
        marker=dict(
            size=12,
            symbol="star"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=[2050],
        y=[prediction_2050_9],
        mode="markers",
        name="2050년 9차 예측",
        marker=dict(
            size=12,
            symbol="star"
        )
    )
)


fig.update_layout(
    title="서울 연평균기온과 다항회귀 모델",
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
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 테스트 구간 확대
# --------------------------------------------------
st.subheader("🔍 테스트 데이터(2005~2025)에서의 예측 비교")

test_fig = go.Figure()

test_fig.add_trace(
    go.Scatter(
        x=test_years,
        y=actual,
        mode="lines+markers",
        name="실제 기온",
        line=dict(width=3)
    )
)

test_fig.add_trace(
    go.Scatter(
        x=test_years,
        y=pred_1,
        mode="lines",
        name="1차 예측",
        line=dict(dash="dash")
    )
)

test_fig.add_trace(
    go.Scatter(
        x=test_years,
        y=pred_3,
        mode="lines",
        name="3차 예측",
        line=dict(dash="dot")
    )
)

test_fig.add_trace(
    go.Scatter(
        x=test_years,
        y=pred_9,
        mode="lines",
        name="9차 예측",
        line=dict(dash="dashdot")
    )
)

test_fig.update_layout(
    title="학습에 사용하지 않은 테스트 데이터에서의 예측",
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
    test_fig,
    use_container_width=True
)


# --------------------------------------------------
# 2050년 예측 비교
# --------------------------------------------------
st.subheader("🔮 2050년 예측 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "1차",
        f"{prediction_2050_1:.2f} °C"
    )

with col2:
    st.metric(
        "3차",
        f"{prediction_2050_3:.2f} °C"
    )

with col3:
    st.metric(
        "9차",
        f"{prediction_2050_9:.2f} °C"
    )


# --------------------------------------------------
# 계산 방법 설명
# --------------------------------------------------
st.subheader("ℹ️ 계산 방법")

st.write(
    "고차 다항회귀에서 연도 자체를 그대로 사용하면 숫자가 커져 "
    "계산이 불안정해질 수 있으므로, 회귀 계산에서는 "
    "**(연도 - 2000) / 100**으로 변환했습니다."
)

st.write(
    "예를 들어 1900년은 -1, 2000년은 0, "
    "2050년은 0.5로 변환하여 계산합니다."
)

st.write(
    "모델은 **2004년까지의 훈련 데이터만 사용**하여 만들었으며, "
    "**2005~2025년 데이터는 모델 학습에 사용하지 않고 오직 평가에만 사용**했습니다."
)

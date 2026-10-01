import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연도별 평균기온을 바탕으로 미래의 연평균기온을 예측합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.stop()

# --------------------------------------------------
# 연도별 데이터 만들기
# 조건:
# 1. 2025년 이후 제외
# 2. 관측일이 300일 미만인 해 제외
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
# 회귀분석
# 독립변수 = 1908년부터 지난 연수
# x = 연도 - 1908
# y = 연평균기온
# --------------------------------------------------

yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀식에 따른 예측값
yearly["회귀예측기온"] = slope * yearly["경과연수"] + intercept

# --------------------------------------------------
# 회귀선 계산
# 1908 ~ 2100
# --------------------------------------------------

future_years = np.arange(1908, 2101)
future_elapsed = future_years - 1908
future_pred = slope * future_elapsed + intercept

# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_elapsed = selected_year - 1908
predicted_temp = slope * selected_elapsed + intercept

# --------------------------------------------------
# 선택한 연도의 예상 기온
# --------------------------------------------------

st.subheader(f"📌 {selected_year}년 예상 연평균기온")

st.metric(
    label="회귀모형 예상 평균기온",
    value=f"{predicted_temp:.2f} °C"
)

# 실제 데이터가 있는 연도라면 실제 평균기온도 표시
actual = yearly[yearly["연도"] == selected_year]

if not actual.empty:
    actual_temp = actual.iloc[0]["연평균기온"]

    st.info(
        f"{selected_year}년 실제 연평균기온: "
        f"{actual_temp:.2f} °C "
        f"(관측일수 {int(actual.iloc[0]['관측일수'])}일)"
    )

# --------------------------------------------------
# 회귀 분석 정보
# --------------------------------------------------

st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{len(yearly)}개"
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

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

st.write(
    f"회귀식: **연평균기온 = {slope:.4f} × (연도 - 1908) "
    f"+ {intercept:.4f}**"
)

# --------------------------------------------------
# Plotly 산점도 + 회귀선
# --------------------------------------------------

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        text=[
            f"{year}년<br>"
            f"연평균기온: {temp:.2f}°C<br>"
            f"관측일수: {days}일"
            for year, temp, days
            in zip(
                yearly["연도"],
                yearly["연평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=future_years,
        y=future_pred,
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "%{x}년<br>"
            "회귀 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 예측점
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(
            size=16,
            symbol="star"
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 연평균기온: {predicted_temp:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 회귀선",
    xaxis=dict(
        title="연도",
        tickmode="linear",
        dtick=10,
        range=[1900, 2100]
    ),
    yaxis=dict(
        title="연평균기온 (°C)"
    ),
    hovermode="closest",
    height=650,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 간단한 설명
# --------------------------------------------------

st.subheader("🔎 데이터 기준")

st.write(
    "• 2025년까지의 데이터만 사용했습니다."
)
st.write(
    "• 한 해의 관측일수가 300일 미만인 연도는 제외했습니다."
)
st.write(
    "• 회귀분석에서는 1908년을 기준으로 해당 연도까지 지난 연수를 "
    "독립변수로 사용했습니다."
)
st.write(
    "• 슬라이더에서 1900~2100년을 선택하면 회귀선에 따른 "
    "예상 연평균기온을 확인할 수 있습니다."
)

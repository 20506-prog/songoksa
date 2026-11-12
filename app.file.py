import pickle
from pathlib import Path
import numpy as np
import streamlit as st
from Orange.data import Domain, Table

# 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통안전 예측 신호등",
    page_icon="🚦",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "traffic_casualty_model.pkcls"


@st.cache_resource
def load_orange_model(model_path: Path):
    """Orange3에서 저장한 .pkcls 모델 파일을 캐싱하여 로드합니다."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_casualties(
    model,
    sido: str,
    accidents: float,
    minor_injuries: float,
    serious_injuries: float | None = None,
    reported_injuries: float | None = None,
) -> float:
    """사용자 입력값을 Orange3 Table로 변환하여 예상 사망자 수를 예측합니다."""
    base_domain = (
        model.original_domain
        if getattr(model, "original_domain", None) is not None
        else model.domain
    )
    input_domain = Domain(base_domain.attributes)

    # 모델의 입력 피처 순서에 맞춰 값 매핑 (미입력 변수는 np.nan 전달 시 모델 내장 평균값으로 자동 대체)
    value_map = {
        "시도": sido,
        "사고건수": float(accidents),
        "중상자수": float(serious_injuries) if serious_injuries is not None else np.nan,
        "경상자수": float(minor_injuries),
        "부상신고자수": float(reported_injuries) if reported_injuries is not None else np.nan,
    }

    row = []
    for attr in input_domain.attributes:
        if attr.name in value_map:
            row.append(value_map[attr.name])
        else:
            row.append(attr.values[0] if attr.is_discrete else np.nan)

    input_table = Table.from_list(input_domain, [row])
    prediction = float(model(input_table)[0])
    return max(0.0, prediction)


def get_traffic_light_status(predicted_deaths: float, safe_limit: float, caution_limit: float):
    """예상 사망자 수에 따른 신호등 상태 및 어린이 보호구역 안전 수칙을 반환합니다."""
    if predicted_deaths < safe_limit:
        return {
            "level": "SAFE",
            "badge": "🟢 안전",
            "title": "안전 단계 — 현재 교통사고 위험도가 낮은 수준입니다",
            "color": "#16a34a",
            "bg": "#f0fdf4",
            "border": "#86efac",
            "desc": "예상 사망자 수가 기준치 미만으로 양호한 상태입니다. 기본 안전 수칙을 준수하며 예방 활동을 유지하세요.",
            "rules": [
                ("시속 30km 이하 서행 유지", "어린이 보호구역 진입 시 규정 속도를 준수하고 전방을 주시합니다."),
                ("횡단보도 앞 일시정지", "신호등이 없는 횡단보도에서도 보행자 유무와 관계없이 일시정지합니다."),
                ("사각지대 보행자 확인", "주정차 차량 사이나 골목길 진입 시 어린이 돌발 횡단에 대비합니다."),
                ("어린이 통학버스 보호", "통학버스 정차 시 일시정지 후 안전을 확인하고 서행으로 통과합니다."),
            ],
        }
    elif predicted_deaths < caution_limit:
        return {
            "level": "CAUTION",
            "badge": "🟡 주의",
            "title": "주의 단계 — 사고 피해 확산 가능성이 감지되었습니다",
            "color": "#ca8a04",
            "bg": "#fefce8",
            "border": "#fde047",
            "desc": "사고건수 및 부상자 규모 증가로 사망사고 전환 위험이 높아지고 있습니다. 등·하교 시간대 집중 관리가 필요합니다.",
            "rules": [
                ("등·하교 시간대 감속 강화", "오전 8~9시, 오후 1~4시 집중 통행 시간에는 시속 20km 이하로 서행합니다."),
                ("불법 주·정차 즉시 계도", "운전자 시야를 가리는 보호구역 내 불법 주정차 차량을 집중 단속합니다."),
                ("교차로 우회전 일시정지 철저", "우회전 시 보행 신호와 횡단보도 주변 어린이를 반드시 확인합니다."),
                ("안전 펜스·반사경 점검", "파손된 방호울타리와 시야 확보용 반사경을 즉시 보수합니다."),
            ],
        }
    else:
        return {
            "level": "DANGER",
            "badge": "🔴 위험",
            "title": "위험 단계 — 중대 교통사고 발생 위험이 매우 높습니다",
            "color": "#dc2626",
            "bg": "#fef2f2",
            "border": "#fca5a5",
            "desc": "예상 사망자 수가 위험 임계치를 초과했습니다. 즉각적인 교통 통제, 시설 보완 및 집중 단속이 요구됩니다.",
            "rules": [
                ("과속·신호위반 무인단속 강화", "이동식·고정식 단속 장비를 가동하고 경찰·녹색어머니회 합동 배치를 시행합니다."),
                ("차량 통행 제한 및 우회 유도", "위험 구간 내 대형 화물차 및 통과 차량의 진입 제한을 검토합니다."),
                ("과속방지턱·노면표시 긴급 정비", "운전자 시인성을 높이는 옐로카펫, 적색 미끄럼방지 포장을 즉시 보강합니다."),
                ("보행자 우선 신호체계 전환", "어린이 보행 시간 연장과 전 방향 적색신호(All-Red) 도입을 추진합니다."),
            ],
        }


def render_traffic_light_html(level: str) -> str:
    """대형 3색 신호등 UI HTML을 생성합니다."""
    red_active = level == "DANGER"
    yellow_active = level == "CAUTION"
    green_active = level == "SAFE"

    red_style = (
        "background: radial-gradient(circle at 35% 35%, #ff6b6b, #dc2626); box-shadow: 0 0 35px #ef4444, 0 0 70px rgba(239,68,68,0.6);"
        if red_active
        else "background: #3f1d1d; opacity: 0.35;"
    )
    yellow_style = (
        "background: radial-gradient(circle at 35% 35%, #fde047, #eab308); box-shadow: 0 0 35px #facc15, 0 0 70px rgba(250,204,21,0.6);"
        if yellow_active
        else "background: #3f3818; opacity: 0.35;"
    )
    green_style = (
        "background: radial-gradient(circle at 35% 35%, #4ade80, #16a34a); box-shadow: 0 0 35px #22c55e, 0 0 70px rgba(34,197,94,0.6);"
        if green_active
        else "background: #143622; opacity: 0.35;"
    )

    return f"""
    <div style="display:flex; justify-content:center; align-items:center; padding: 12px 0;">
        <div style="
            background: linear-gradient(145deg, #1f2937, #111827);
            border: 4px solid #374151;
            border-radius: 36px;
            padding: 24px 28px;
            display: flex;
            flex-direction: column;
            gap: 22px;
            box-shadow: 0 18px 35px rgba(0,0,0,0.35);
            width: 170px;
            align-items: center;
        ">
            <div style="width: 105px; height: 105px; border-radius: 50%; border: 3px solid #111827; transition: all 0.3s ease; {red_style}"></div>
            <div style="width: 105px; height: 105px; border-radius: 50%; border: 3px solid #111827; transition: all 0.3s ease; {yellow_style}"></div>
            <div style="width: 105px; height: 105px; border-radius: 50%; border: 3px solid #111827; transition: all 0.3s ease; {green_style}"></div>
        </div>
    </div>
    """


# --- 메인 화면 구성 ---
st.title("🚦 어린이 보호구역 교통사고 사망자 예측 및 안전 신호등")
st.caption("Orange3 학습 모델(`traffic_casualty_model.pkcls`) 기반 실시간 위험도 시뮬레이터")

if not MODEL_PATH.exists():
    st.error(f"모델 파일을 찾을 수 없습니다: `{MODEL_PATH.name}` 파일을 `app.py`와 같은 폴더에 배치해주세요.")
    st.stop()

model = load_orange_model(MODEL_PATH)

# 모델의 시도 목록 자동 추출 (기본값: ['경기', '부산', '서울'])
sido_options = ["서울", "경기", "부산"]
base_domain = getattr(model, "original_domain", None) or model.domain
for attr in base_domain.attributes:
    if attr.name == "시도" and attr.is_discrete:
        sido_options = list(attr.values)
        break

# 사이드바: 신호등 임계값 설정
with st.sidebar:
    st.header("⚙️ 신호등 판정 기준 설정")
    safe_limit = st.number_input("🟢 안전 기준 (예상 사망자 수 미만)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
    caution_limit = st.number_input("🟡 주의 기준 (예상 사망자 수 미만)", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
    if caution_limit <= safe_limit:
        st.warning("주의 기준은 안전 기준보다 커야 합니다.")
        caution_limit = safe_limit + 5.0

    st.divider()
    st.markdown("**모델 정보**")
    st.write(f"- 모델 유형: `{type(model).__name__}`")
    st.write(f"- 선택 가능 시도: `{', '.join(sido_options)}`")

# 1. 입력 컨트롤 영역
st.subheader("1. 지역 및 사고 지표 입력")
col_in1, col_in2, col_in3 = st.columns([1, 1.5, 1.5])

with col_in1:
    selected_sido = st.selectbox("시도 선택", options=sido_options, index=0)

with col_in2:
    accidents_val = st.slider(
        "사고건수 (건)",
        min_value=0,
        max_value=4000,
        value=1000,
        step=10,
        help="슬라이더를 움직이면 예상 사망자 수가 즉시 재계산됩니다.",
    )

with col_in3:
    minor_injuries_val = st.slider(
        "경상자수 (명)",
        min_value=0,
        max_value=4000,
        value=1000,
        step=10,
        help="슬라이더를 움직이면 예상 사망자 수가 즉시 재계산됩니다.",
    )

# 선택 사항: 모델에 포함된 나머지 2개 변수(중상자수, 부상신고자수) 직접 조절 옵션
with st.expander("🔧 추가 변수 세부 조절 (미조절 시 학습 데이터 평균값 자동 적용)"):
    use_custom_extra = st.checkbox("중상자수·부상신고자수 직접 입력하기", value=False)
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        serious_val = st.slider("중상자수 (명)", min_value=0, max_value=1000, value=318, step=5, disabled=not use_custom_extra)
    with ex_col2:
        reported_val = st.slider("부상신고자수 (명)", min_value=0, max_value=600, value=145, step=5, disabled=not use_custom_extra)

# 2. 실시간 모델 예측 실행
predicted_deaths = predict_casualties(
    model=model,
    sido=selected_sido,
    accidents=accidents_val,
    minor_injuries=minor_injuries_val,
    serious_injuries=serious_val if use_custom_extra else None,
    reported_injuries=reported_val if use_custom_extra else None,
)

status = get_traffic_light_status(predicted_deaths, safe_limit, caution_limit)

st.divider()

# 3. 예측 결과 및 대형 신호등 + 안전 수칙 카드 영역
st.subheader("2. 실시간 예측 결과 및 어린이 보호구역 안전 가이드")
res_col1, res_col2 = st.columns([1, 2.2])

with res_col1:
    st.markdown(render_traffic_light_html(status["level"]), unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center; font-size:1.5rem; font-weight:800; color:{status['color']}; margin-top:4px;'>{status['badge']}</div>",
        unsafe_allow_html=True,
    )

with res_col2:
    # 예측 지표 카드
    st.markdown(
        f"""
        <div style="
            background-color: {status['bg']};
            border: 2px solid {status['border']};
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 0.95rem; color: #4b5563; font-weight: 600;">
                [{selected_sido}] 사고 {accidents_val:,}건 · 경상자 {minor_injuries_val:,}명 기준
            </div>
            <div style="font-size: 2.3rem; font-weight: 900; color: {status['color']}; margin: 6px 0;">
                예상 사망자 수: {predicted_deaths:.2f} 명
                <span style="font-size: 1.05rem; font-weight: 600; color: #374151;">(약 {round(predicted_deaths)}명)</span>
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-bottom: 4px;">
                {status['title']}
            </div>
            <div style="font-size: 0.95rem; color: #374151;">
                {status['desc']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 어린이 보호구역 안전 수칙 카드 2x2 그리드
    st.markdown("#### 🛡️ 어린이 보호구역 핵심 안전 수칙 카드")
    card_cols = st.columns(2)
    for idx, (rule_title, rule_body) in enumerate(status["rules"]):
        with card_cols[idx % 2]:
            st.markdown(
                f"""
                <div style="
                    background-color: #ffffff;
                    border: 1px solid #e5e7eb;
                    border-left: 5px solid {status['color']};
                    border-radius: 12px;
                    padding: 14px 16px;
                    margin-bottom: 12px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                ">
                    <div style="font-weight: 700; font-size: 1.0rem; color: #111827; margin-bottom: 4px;">
                        {idx + 1}. {rule_title}
                    </div>
                    <div style="font-size: 0.9rem; color: #4b5563; line-height: 1.45;">
                        {rule_body}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
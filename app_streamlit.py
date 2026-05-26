from pathlib import Path

import pandas as pd
import streamlit as st

from elderly_risk_system import ElderlyRiskSystem


st.set_page_config(page_title="老年慢病智能预警系统", layout="wide")
st.title("面向老年慢病的大数据挖掘及智能健康风险预警系统")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = ""
if "username" not in st.session_state:
    st.session_state.username = ""

USER_DB = {
    "elder_001": {"password": "elder123", "role": "老人"},
    "care_001": {"password": "care123", "role": "照护者"},
    "doctor_001": {"password": "doctor123", "role": "医生"},
}

with st.sidebar:
    if not st.session_state.logged_in:
        st.header("登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        login_btn = st.button("登录", type="primary")
        if login_btn:
            if username in USER_DB and USER_DB[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role = USER_DB[username]["role"]
                st.session_state.username = username
                st.success("登录成功")
                st.rerun()
            else:
                st.error("用户名或密码错误")
    else:
        st.success(f"已登录：{st.session_state.username}（{st.session_state.role}）")
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.session_state.role = ""
            st.session_state.username = ""
            st.rerun()

if not st.session_state.logged_in:
    st.info("请先在左侧登录。测试账号：elder_001 / care_001 / doctor_001")
    st.stop()

with st.sidebar:
    st.header("系统参数")
    data_dir = st.text_input("数据目录", value=".")
    out_dir = st.text_input("输出目录", value="./outputs")
    run_train = st.button("1) 训练并生成预警")
    load_existing = st.button("2) 加载已有结果")

system = ElderlyRiskSystem(data_dir=data_dir, output_dir=out_dir)
output_path = Path(out_dir).resolve()

all_risk = None
if run_train:
    with st.spinner("正在融合三部分数据并训练模型..."):
        result = system.train()
        all_risk = system.predict_all_users()
        overview = system.risk_overview()
    st.success("训练和预警生成完成")
    c1, c2, c3 = st.columns(3)
    c1.metric("样本数", result.sample_count)
    c2.metric("AUC", f"{result.auc:.4f}")
    c3.metric("F1", f"{result.f1:.4f}")
    st.subheader("全体用户预警结果")
    st.dataframe(all_risk, use_container_width=True)
    st.subheader("总体概览")
    st.json(overview)
elif load_existing:
    risk_file = output_path / "all_user_risk.csv"
    test_file = output_path / "test_predictions.csv"
    if not risk_file.exists():
        st.error("未找到 all_user_risk.csv，请先训练。")
    else:
        all_risk = pd.read_csv(risk_file)
        st.subheader("全体用户预警结果")
        st.dataframe(all_risk, use_container_width=True)
        if test_file.exists():
            st.subheader("测试集预测结果")
            st.dataframe(pd.read_csv(test_file), use_container_width=True)
else:
    risk_file = output_path / "all_user_risk.csv"
    if risk_file.exists():
        all_risk = pd.read_csv(risk_file)
        st.subheader("全体用户预警结果")
        st.dataframe(all_risk, use_container_width=True)
    else:
        st.info("先在左侧点击“训练并生成预警”")

st.markdown("---")
st.subheader("单个用户详情页")
if all_risk is not None and not all_risk.empty:
    user_ids = all_risk["user_id"].astype(str).tolist()
    selected_user = st.selectbox("选择用户", user_ids)
    if st.button("查看用户详情", type="primary"):
        try:
            detail = system.get_user_detail(selected_user)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("风险概率", f"{detail['risk_prob']:.3f}")
            c2.metric("当前风险等级", detail["risk_level"])
            c3.metric("90天预警", detail["window_90d"])
            c4.metric("180天预警", detail["window_180d"])
            st.caption(f"360天预警：{detail['window_360d']}")

            st.write(f"用户信息：{detail['user_id']} / {detail['sex']} / {detail['age_group']}")

            metric_df = pd.DataFrame(
                [{"指标": k, "值": v} for k, v in detail["key_metrics"].items()]
            )
            st.markdown("**关键指标**")
            st.dataframe(metric_df, use_container_width=True)

            factor_df = pd.DataFrame(detail["top_factors"])
            st.markdown("**主要风险因素（近似解释）**")
            st.dataframe(factor_df, use_container_width=True)

            st.markdown("**干预建议**")
            for idx, suggestion in enumerate(detail["suggestions"], start=1):
                st.write(f"{idx}. {suggestion}")
        except Exception as exc:
            st.error(f"加载用户详情失败：{exc}")
else:
    st.info("暂无可用风险结果，请先训练或加载结果。")

st.markdown("---")
st.caption(
    "数据使用范围：Demographics + Sleep_physio/Sleep_state + ScanWatch_HR/Steps（已全部纳入）"
)

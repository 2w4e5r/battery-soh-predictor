import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="蓄电池SOH预测", layout="wide")
st.title("🔋 蓄电池健康状态（SOH）预测系统")
st.markdown("基于随机森林回归的电池退化预测")

@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

feature_cols = ['Voltage', 'Current', 'Temperature', 'Capacity',
                'Voltage_mean_last3', 'Voltage_std_last3',
                'Current_mean_last3', 'Current_std_last3',
                'Temperature_mean_last3', 'Temperature_std_last3',
                'Capacity_decay_rate', 'Voltage_diff']

with st.sidebar:
    st.header("📘 使用说明")
    st.write("""
    - **单点预测**：手动输入电池当前特征值，实时预测 SOH。
    - **批量预测**：上传 CSV 文件（必须包含上述 12 个特征列），自动添加预测结果并下载。
    """)

st.header("📊 模型性能")
try:
    st.image('prediction_scatter.png', caption='测试集预测对比图')
except:
    st.write("未找到散点图文件，请先运行 train_model.py 生成。")

st.header("🔮 单点预测")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        voltage = st.number_input("电压 (V)", value=3.5, step=0.01)
        current = st.number_input("电流 (A)", value=4.8, step=0.1)
        temperature = st.number_input("温度 (℃)", value=26.0, step=0.1)
        capacity = st.number_input("容量 (Ah)", value=1.9, step=0.01)
    with col2:
        volt_mean = st.number_input("最近3次电压均值 (V)", value=3.52)
        volt_std = st.number_input("最近3次电压标准差", value=0.02)
        curr_mean = st.number_input("最近3次电流均值 (A)", value=4.75)
        curr_std = st.number_input("最近3次电流标准差", value=0.05)
        temp_mean = st.number_input("最近3次温度均值 (℃)", value=26.2)
        temp_std = st.number_input("最近3次温度标准差", value=0.3)
        cap_decay = st.number_input("容量衰减率", value=0.95, step=0.01)
        volt_diff = st.number_input("电压变化差值 (V)", value=-0.01, step=0.01)
    submitted = st.form_submit_button("🚀 预测 SOH")
    if submitted:
        input_data = np.array([[voltage, current, temperature, capacity,
                                volt_mean, volt_std, curr_mean, curr_std,
                                temp_mean, temp_std, cap_decay, volt_diff]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        st.success(f"📈 预测的电池健康状态 (SOH) 为: **{pred:.2f}%**")

st.header("📂 批量预测（上传 CSV 文件）")
uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        if all(col in df_upload.columns for col in feature_cols):
            X_upload = df_upload[feature_cols]
            X_upload_scaled = scaler.transform(X_upload)
            preds = model.predict(X_upload_scaled)
            df_upload['Predicted_SOH'] = preds
            st.dataframe(df_upload)
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button("下载预测结果", csv, "predicted_soh.csv", "text/csv")
        else:
            st.error(f"上传文件缺少必要特征列，需要包含：{', '.join(feature_cols)}")
    except Exception as e:
        st.error(f"文件处理出错：{e}")
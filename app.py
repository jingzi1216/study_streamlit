import streamlit as st
import joblib
import numpy as np

# 加载模型
model = joblib.load('RFC.pkl')


# 类别特征到数值的映射     注：在Streamlit应用中，用户输入的是中文字符串，而不是编码后的数值。
cp_options = {
    "典型胸痛": 0,
    "非典型胸痛": 1,
    "非胸痛": 2
}

restecg_options = {
    "正常": 0,
    "ST-T波异常": 1,
    "左心室肥大": 2
}

slope_options = {
    "上升": 1,
    "平坦": 2,
    "下降": 3
}

thal_options = {
    "正常": 1,
    "固定缺陷": 2,
    "可逆缺陷": 3
}

# Streamlit应用界面
st.title('心脏病预测器')
st.write('请输入以下信息以预测心脏病风险：')

# 用户输入
age = st.number_input("年龄")
sex = st.selectbox("性别 (0=女性, 1=男性)", options=[0, 1], format_func=lambda x: '女性' if x == 0 else '男性')
cp = st.selectbox("胸痛类型", options=list(cp_options.keys()), format_func=lambda x: x)
trestbps = st.number_input("静息血压")
chol = st.number_input("血清胆固醇")
fbs = st.selectbox("空腹血糖>120 mg/dl", options=[0, 1], format_func=lambda x: '否' if x == 0 else '是')
restecg = st.selectbox("静息心电图结果", options=list(restecg_options.keys()), format_func=lambda x: x)
thalach = st.number_input("最大心率")
exang = st.selectbox("运动诱发心绞痛", options=[0, 1], format_func=lambda x: '否' if x == 0 else '是')
oldpeak = st.number_input("运动相对静息ST段下降")
slope = st.selectbox("运动峰值ST段斜率", options=list(slope_options.keys()), format_func=lambda x: x)
ca = st.number_input("荧光透视显示的主要血管数")
thal = st.selectbox("Thal", options=list(thal_options.keys()), format_func=lambda x: x)

# 将类别特征转换为数值
cp_value = cp_options[cp]
restecg_value = restecg_options[restecg]
slope_value = slope_options[slope]
thal_value = thal_options[thal]

# 构建特征数组
features = np.array([[age, sex, cp_value, trestbps, chol, fbs, restecg_value, thalach, exang, oldpeak, slope_value, ca, thal_value]])

# 预测
if st.button("预测"):
    prediction = model.predict(features)
    st.write(f"预测结果: {'有心脏病' if prediction[0] == 1 else '无心脏病'}")
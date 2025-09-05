import streamlit as st
import pandas as pd
import numpy as np
import LinearRegression
from sklearn.linear_model 


# Khởi tạo session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Tiêu đề app
st.title("📈 Dự đoán giá trị bằng Linear Regression")

# Upload dữ liệu
uploaded_file = st.file_uploader("📤 Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.data = df
    st.write("📊 Dữ liệu đã upload:")
    st.dataframe(df)

# Chọn biến đầu vào và đầu ra
if st.session_state.data is not None:
    df = st.session_state.data
    columns = df.columns.tolist()

    st.subheader("🔧 Chọn biến đầu vào và đầu ra")
    input_features = st.multiselect("Chọn biến đầu vào (X)", columns)
    target_feature = st.selectbox("Chọn biến đầu ra (y)", columns)

    if input_features and target_feature:
        X = df[input_features]
        y = df[target_feature]

        # Huấn luyện mô hình
        if st.button("🏋️ Huấn luyện mô hình"):
            model = LinearRegression()
            model.fit(X, y)
            st.session_state.model = model
            st.session_state.model_trained = True
            st.success("✅ Mô hình đã được huấn luyện thành công!")

        # Dự đoán với dữ liệu mới
        if st.session_state.model_trained:
            st.subheader("🔍 Dự đoán với dữ liệu mới")
            input_values = []
            for feature in input_features:
                value = st.number_input(f"Nhập giá trị cho {feature}", value=0.0)
                input_values.append(value)

            if st.button("📌 Dự đoán"):
                input_array = np.array(input_values).reshape(1, -1)
                prediction = st.session_state.model.predict(input_array)
                st.success(f"📈 Giá trị dự đoán: {prediction[0]:.2f}")

            # Hiển thị thông tin mô hình
            st.subheader("📋 Thông tin mô hình")
            coef_df = pd.DataFrame({
                "Biến đầu vào": input_features,
                "Hệ số hồi quy": st.session_state.model.coef_
            })
            st.dataframe(coef_df)

            r2_score = st.session_state.model.score(X, y)
            st.write(f"🔎 Độ chính xác mô hình (R²): {r2_score:.2f}")

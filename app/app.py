from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import numpy as np
import streamlit as st
from PIL import Image

icon = Image.open("app/img/anya.png")
st.set_page_config(
    page_title="DOG - CAT PREDICTION",
    page_icon=icon,
)

class_names = ["Cat", "Dog"]

mobilenetv2_model = load_model('./src/models/mobilenetv2_feature_extractor.h5')
pca = joblib.load('./src/models/pca_transformer.joblib')
SVM = joblib.load('./src/models/SVM.joblib')

target_size = (224, 224)


# Sidebar
with st.sidebar:
    st.image(icon)
    st.subheader("Dog cat prediction with mobilenetv2 + SVM")
    st.caption("=== Nguyen Hai Nam ===")

    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose dog or cat image")


# Body
st.header("DOG CAT PREDICTION")

col1, col2 = st.columns(2)
y_pred = None

if uploaded_file is not None:
    with col1:
        st.subheader(":camera: Input")
        st.image(uploaded_file, use_column_width=True)
        img = tf.keras.preprocessing.image.load_img(
            uploaded_file, target_size=target_size
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img_aux = img.copy()


        if st.button(
            ":arrows_counterclockwise: Predict Dog or Cat "
        ):
            features = mobilenetv2_model.predict(img)
            features = pca.transform(features)
            with st.spinner("Wait for it..."):
                y_pred = SVM.predict(features)


            st.subheader(":white_check_mark: Prediction")

            st.metric(
                label="Type:",
                value=f"{class_names[int(y_pred)]}",
            )
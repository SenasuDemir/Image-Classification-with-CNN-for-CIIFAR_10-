import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

model = load_model('cnn_model.h5')

def process_image(img):
    img = img.resize((32, 32))  
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

st.markdown("""
    <style>
        .title {
            font-size: 50px;
            color: #A8D5FF;  # Pastel Blue color
            font-weight: bold;
            text-align: center;
        }
        .description {
            font-size: 16px;
            color: #555555;
            text-align: center;
        }
        .uploaded-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            padding: 20px;
        }
        .prediction {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .confidence {
            font-size: 18px;
            text-align: center;
            color: #4CAF50;
        }
        .bar-chart {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">CIFAR-10 Image Classification 🔎</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image to classify it into one of the CIFAR-10 categories.</div>', unsafe_allow_html=True)

file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
    st.image(img, caption='Uploaded Image', use_column_width=True, output_format='PNG')
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner('Processing...'):
        image = process_image(img)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        confidence = np.max(predictions) * 100  # Confidence score

        st.markdown(f'<div class="prediction">Prediction: {class_names[predicted_class]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="bar-chart">Class Probabilities:</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame(predictions[0], index=class_names, columns=["Probability"])
        st.bar_chart(prob_df)


st.markdown('---')
st.markdown("<div style='text-align: center;'>Powered by Streamlit and TensorFlow</div>", unsafe_allow_html=True)

"""
Streamlit IMDB Sentiment Analyzer (Simplified + Visualization)
--------------------------------------------------------------
• Clean, minimal UI
• Error handling for model loading and prediction
• Adds sentiment classification scale and visual
"""

import time
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 IMDB Sentiment Analyzer")
st.caption("Classify movie reviews as positive, negative, or mixed with a pre-trained RNN")

# -----------------------------------------------------------------------------
# Cached assets
# -----------------------------------------------------------------------------

NUM_WORDS = 10_000
MAX_LEN = 500

@st.cache_resource(show_spinner=False)
def load_word_index():
    word_index = imdb.get_word_index()
    return word_index

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    try:
        model = load_model("simple_rnn_imdb.keras")
        return model
    except Exception as e:
        st.error("❌ Model loading failed!")
        st.exception(e)
        st.stop()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def preprocess_text(text: str, word_index: dict) -> np.ndarray:
    tokens = text.lower().split()

    encoded = []

    for word in tokens:
        index = word_index.get(word)

        if index is None:
            index = 2  # Unknown word
        else:
            index = index + 3  # IMDB reserved token offset

        if index < NUM_WORDS:
            encoded.append(index)
        else:
            encoded.append(2)

    return sequence.pad_sequences(
        [encoded],
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre"
    )

def classify_sentiment(score: float) -> str:
    if score < 0.001:
        return "😊 Mixed"
    elif score > 0.02:
        return "🙁 Negative"
    elif:
        return "😐 Positive"

def sentiment_visual(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': '#ff9999'},
                {'range': [40, 60], 'color': '#ffd480'},
                {'range': [60, 100], 'color': '#66ff99'}
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score (%)"},
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=0, l=0, r=0))
    return fig

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        "* **Model:** Simple RNN with ReLU activation\n"
        "* **Training data:** 50 000 IMDB reviews\n"
        "* **Task:** Binary sentiment classification\n"
        "* **Tech:** TensorFlow/Keras • Streamlit • NumPy"
    )

# -----------------------------------------------------------------------------
# Main interaction
# -----------------------------------------------------------------------------
word_index = load_word_index()
model = load_sentiment_model()

sample_reviews = {
    "— Select a sample —": "",
    "Positive example": "This movie was absolutely fantastic! The acting was superb and the plot was gripping.",
    "Negative example": "One of the worst movies I've ever seen. The plot made no sense and the acting was terrible.",
    "Mixed example": "Great visuals but the story was shallow and predictable – enjoyable yet forgettable.",
}

sample_key = st.selectbox("Need inspiration?", list(sample_reviews.keys()))
review_text = st.text_area(
    "Write or paste a movie review",
    value=sample_reviews[sample_key],
    height=150,
    placeholder="Enter your movie review here…",
)

if st.button("Analyze sentiment"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review first.")
        st.stop()

    with st.spinner("Analyzing…"):
        time.sleep(0.2)
        try:
            x = preprocess_text(review_text, word_index)
            prediction = model.predict(x, verbose=0)
            st.write("Raw model output:", prediction)

            prob = float(prediction[0][0])
        except Exception as e:
            st.error("❌ Prediction failed. Please try again with different text.")
            st.exception(e)
            st.stop()

    st.subheader(classify_sentiment(prob))
    st.plotly_chart(sentiment_visual(prob), use_container_width=True)

    with st.expander("Details"):
        st.write("**Model raw score:**", f"{prob:.4f}")
        st.write("**Words in review:**", len(review_text.split()))
        st.write("**Analyzed at:**", datetime.now().strftime("%H:%M:%S"))

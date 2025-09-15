import streamlit as st
import joblib
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        # Check if vectorizer is fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            st.error("Vectorizer is not fitted. Please provide a fitted vectorizer.")
            return None, None
        
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please ensure 'random_forest.joblib' and 'tfidf_vectorizer.joblib' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and vectorizer
model, vectorizer = load_model()

# Define label mapping
label_mapping = {
    'TRUE': 'Real News ✅',
    'FALSE': 'Fake News ❌',
    'half-true': 'Partially True ⚠️',
    'mostly-true': 'Mostly True 🔍',
    'pants-fire': 'False (Pants on Fire) 🔥',
    'barely-true': 'Barely True ⚡'
}

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 900;
        background: linear-gradient(45deg, #FF4B4B, #FF9B50, #FFD166, #4ECDC4, #556270);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 8s ease infinite;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2B2B2B;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF4B4B, #FF9B50);
        color: white;
    }
    
    /* Sub header styling */
    .sub-header {
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #4ECDC4, #556270);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(135deg, #2C3E50 0%, #4A569D 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.25);
        color: white;
        text-align: center;
        margin-top: 2rem;
        border: 3px solid rgba(255,255,255,0.1);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 86, 157, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(76, 86, 157, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 86, 157, 0); }
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF9B50);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: 800;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 8px 15px rgba(255, 75, 75, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 20px rgba(255, 75, 75, 0.6);
    }
    
    /* Input box styling */
    .input-box {
        background: linear-gradient(135deg, #1E1E1E 0%, #2B2B2B 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        border: 2px solid rgba(255,255,255,0.05);
    }
    
    /* Text input styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2B2B2B;
        color: white;
        border-radius: 15px;
        padding: 1.2rem;
        border: 2px solid #444;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #4ECDC4;
        box-shadow: 0 0 0 2px rgba(78, 205, 196, 0.2);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(45deg, #FF4B4B, #FF9B50, #FFD166, #4ECDC4);
        border-radius: 10px;
        height: 15px;
    }
    
    /* Card styling */
    .card {
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #FF4B4B;
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Custom metric styling */
    .stMetric {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header floating">📰 FAKE NEWS PREDICTOR 🔍</h1>', unsafe_allow_html=True)

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["🔎 Predict", "ℹ️ About", "📊 Data Stats"])

# Prediction Tab
with tab1:
    st.markdown('<h2 class="sub-header">Check News Authenticity</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-box">', unsafe_allow_html=True)
        news_text = st.text_area(
            "Paste the news content here:",
            height=200,
            placeholder="Enter the news statement you want to verify...",
            help="The more text you provide, the more accurate the prediction will be!"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("Analyze News 🚀", disabled=(model is None or vectorizer is None))
        
        if predict_btn and news_text:
            if model is None or vectorizer is None:
                st.error("Model not loaded. Please check if the model files are available.")
            else:
                with st.spinner("Analyzing content..."):
                    # Create a custom progress bar with more visual appeal
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    for percent_complete in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(percent_complete + 1)
                        progress_text.markdown(
                            f"<div style='text-align: center; color: #FF4B4B; font-weight: bold;'>Processing: {percent_complete + 1}%</div>", 
                            unsafe_allow_html=True
                        )
                    
                    # Transform the input text
                    transformed_text = vectorizer.transform([news_text])
                    
                    # Make prediction
                    prediction = model.predict(transformed_text)
                    prediction_proba = model.predict_proba(transformed_text)
                    
                    # Get the predicted label
                    predicted_label = prediction[0]
                    human_readable_label = label_mapping.get(predicted_label, predicted_label)
                    
                    # Create a confidence gauge
                    max_proba = np.max(prediction_proba)
                    
                    # Display results with enhanced visual effects
                    st.balloons()  # Celebration effect
                    
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    # Dynamic icon based on prediction
                    if "Real" in human_readable_label:
                        icon = "✅"
                        color = "lightgreen"
                    elif "Fake" in human_readable_label or "False" in human_readable_label:
                        icon = "❌"
                        color = "lightcoral"
                    else:
                        icon = "⚠️"
                        color = "yellow"
                    
                    st.markdown(f"<h2 style='color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>Prediction: {human_readable_label} {icon}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color: white;'>Confidence: {max_proba*100:.2f}%</h3>", unsafe_allow_html=True)
                    
                    # Create gauge chart with dynamic colors
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = max_proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Level", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': "lightcoral"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300, 
                        font={'color': "white", 'family': "Arial"},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add some advice based on prediction
                    if max_proba < 0.7:
                        st.info("⚠️ The confidence in this prediction is moderate. Consider verifying this information through additional sources.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
        elif predict_btn and not news_text:
            st.error("🚫 Please enter some news content to analyze!")

# About Tab
with tab2:
    st.markdown('<h2 class="sub-header">About This App</h2>', unsafe_allow_html=True)
    
    # Use columns for a more engaging layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📰 Fake News Detection</h3>
            <p>This app uses machine learning to analyze news content and predict its authenticity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>🔍 How It Works</h3>
            <ol>
                <li>Text is processed using TF-IDF vectorization</li>
                <li>Features are analyzed by a trained Random Forest model</li>
                <li>Results are displayed with confidence levels</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4054/4054617.png", width=150)
        st.markdown("""
        <div style="text-align: center;">
            <h4 style="color: #FF4B4B;">Stay Informed</h4>
            <p>Always verify information from multiple reliable sources</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Label explanations with colorful cards
    st.markdown("### 🎯 Prediction Categories")
    
    cols = st.columns(3)
    label_info = [
        ("✅ Real News", "Information verified as accurate", "#4ECDC4"),
        ("❌ Fake News", "Deliberately false information", "#FF6B6B"),
        ("⚠️ Partially True", "Contains elements of truth but misleading", "#FFD166"),
        ("🔍 Mostly True", "Mostly accurate but minor inaccuracies", "#51CF66"),
        ("🔥 Pants on Fire", "Completely false and ridiculous claims", "#FF4B4B"),
        ("⚡ Barely True", "Contains minimal truth, mostly false", "#F9A826")
    ]
    
    for i, (title, desc, color) in enumerate(label_info):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Data Stats Tab
with tab3:
    st.markdown('<h2 class="sub-header">Dataset Statistics</h2>', unsafe_allow_html=True)
    
    # Create sample data based on the notebook info
    labels = ['TRUE', 'FALSE', 'half-true', 'mostly-true', 'barely-true', 'pants-fire']
    counts = [2053, 2504, 2627, 2454, 2102, 1047]
    colors = ['#4ECDC4', '#FF6B6B', '#FFD166', '#51CF66', '#F9A826', '#FF4B4B']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=counts, 
            names=labels, 
            title="<b>Distribution of News Labels</b>",
            color_discrete_sequence=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=labels, 
            y=counts, 
            title="<b>Count of Each Label Type</b>",
            labels={'x': 'Label', 'y': 'Count'},
            color=labels,
            color_discrete_sequence=colors
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Text length visualization
    st.markdown("### 📈 Text Length Statistics")
    
    # Generate some sample text length data
    np.random.seed(42)
    text_lengths = np.random.normal(107, 50, 1000)
    text_lengths = [max(11, min(3192, int(length))) for length in text_lengths]
    
    fig = px.histogram(
        x=text_lengths, 
        title="<b>Distribution of Text Lengths</b>",
        labels={'x': 'Text Length (characters)'},
        color_discrete_sequence=['#FF4B4B'],
        nbins=30
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        bargap=0.1
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add some statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Length", f"{np.mean(text_lengths):.0f} chars")
    col2.metric("Minimum Length", f"{np.min(text_lengths)} chars")
    col3.metric("Maximum Length", f"{np.max(text_lengths)} chars")

# Footer with enhanced design
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem; background: linear-gradient(45deg, #1E1E1E, #2B2B2B); color: white; border-radius: 15px;'>
        <h3 style='color: white; margin-bottom: 1rem;'>🔍 Stay Informed, Stay Safe</h3>
        <p>This app uses machine learning to detect fake news. Always verify critical information from multiple reliable sources.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
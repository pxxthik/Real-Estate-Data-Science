import streamlit as st
from utils.css import inject_css

# Constants
PAGE_CONFIG = {
    "page_title": "Real Estate App",
    "page_icon": "🏠"
}
st.set_page_config(**PAGE_CONFIG)

# CSS
inject_css()

# Components
def hero_section():
    """Render the hero section"""
    st.header("Real Estate 🏠 Data Science Application")
    st.markdown("###### Get instant price estimates using our AI-powered valuation tool 🤖")

def disclaimer():
    """Render the disclaimer box"""
    st.info("""⚠️ **Disclaimer:**  
    This application uses 99acres.com data for educational purposes only.""")

def cta_button():
    """Render the CTA button"""
    st.markdown(
        f'<a href="/Price_Predictor" class="cta-link">🚀 Get Instant Estimate →</a>',
        unsafe_allow_html=True
    )

def feature_card(emoji: str, title: str, description: str):
    """Create a feature card component"""
    return f"""
    <div class="feature-card">
        <div style="font-size: 2rem;">{emoji}</div>
        <h5>{title}</h5>
        <p>{description}</p>
    </div>
    """

def metric_box(title: str, items: list):
    """Create a metric box component"""
    items_html = "".join([f"• {item}<br>" for item in items])
    return f"""
    <div class="metric-box">
        <h4>{title}</h4>
        <p style="line-height: 1.6;">{items_html}</p>
    </div>
    """

def profile_link(emoji: str, text: str, url: str):
    """Create a profile link component"""
    return f'''
    <a href="{url}" class="profile-link" target="_blank">
        {emoji} <span>{text}</span>
    </a>
    '''

# Page Sections
def main_features():
    """Render the main features section"""
    st.markdown("### ✨ Main Features")
    with st.container():
        cols = st.columns(3)
        features = [
            ("🏠", "AI Price Predictor", "Instant valuation using machine learning models"),
            ("📊", "Market Analytics", "Interactive charts & trend analysis"),
            ("🤝", "Smart Recommendations", "Personalized apartment suggestions")
        ]
        for col, feature in zip(cols, features):
            with col:
                st.markdown(feature_card(*feature), unsafe_allow_html=True)

def data_section():
    """Render the data & accuracy section"""
    st.markdown("### 📈 Data & Accuracy")
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            content = metric_box(
                "📊 Data Sources",
                [
                    "Trained on 99acres.com property listings",
                    "99acres.com historical data",
                    "Gurgaon-focused market analysis",
                    "Carefully curated local listings"
                ]
            )
            st.markdown(content, unsafe_allow_html=True)
        
        with col2:
            content = metric_box(
                "🎯 Model Approach",
                [
                    "Ensemble learning techniques",
                    "Hyperlocal price factors",
                    "95% Confidence Intervals",
                    "Cross-validated results"
                ]
            )
            st.markdown(content, unsafe_allow_html=True)
    
    st.markdown("""<div style="text-align: center; margin-top: 1rem; color: #666; font-size: 0.9em;">
        🔒 Data used for educational purposes only</div>""", 
        unsafe_allow_html=True)

def sidebar_content():
    """Render the sidebar content"""
    with st.sidebar:
        st.markdown("### Connect with Me 👋")
        links = [
            ("🐙", "GitHub Repository", "https://github.com/pxxthik"),
            ("🎨", "My Portfolio", "https://pratheek-bedre.web.app")
        ]
        for emoji, text, url in links:
            st.markdown(profile_link(emoji, text, url), unsafe_allow_html=True)

# Main Page
def main():
    
    inject_css()
    
    hero_section()
    disclaimer()
    cta_button()
    st.image("assets/real-estate.jpg")
    st.markdown("---")
    
    main_features()
    st.markdown("---")
    
    data_section()
    st.markdown("---")
    
    sidebar_content()

if __name__ == "__main__":
    main()
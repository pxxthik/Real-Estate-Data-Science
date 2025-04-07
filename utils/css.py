import streamlit as st

# CSS Styles
def inject_css():
    """Inject all custom CSS styles"""
    st.markdown("""
    <style>
    /* CTA Link */
    .cta-link {
        display: inline-block;
        padding: 0.8rem 2rem;
        margin: 0 0 1rem 0;
        background: linear-gradient(135deg, #FF4B4B, #FF2B2B);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 12px;
        text-align: center;
        text-decoration: none !important;
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .cta-link:hover {
        background: linear-gradient(135deg, #FF2B2B, #FF1A1A);
        box-shadow: 0 10px 24px rgba(255, 43, 43, 0.4);
        transform: translateY(-2px);
    }
    
    /* Feature Cards */
    .feature-card {
        padding: 1rem;
        height: 14rem;
        border-radius: 12px;
        background: #f8f9fa;
        margin: 0.8rem 0;
        border: 1px solid #eee;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric Boxes */
    .metric-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: #f8f9fa;
        margin: 1rem 0;
        border: 1px solid #eee;
    }
    
    /* Profile Links */
    .profile-link {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem;
        color: #333 !important;
        text-decoration: none !important;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .profile-link:hover {
        background: #FF4B4B10;
        transform: translateX(3px);
    }
    </style>
    """, unsafe_allow_html=True)

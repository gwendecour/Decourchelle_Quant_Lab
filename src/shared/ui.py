import streamlit as st
import os

def render_header():
    """
    Renders a horizontal navigation bar and hides the native sidebar.
    """
    # CSS TO HIDE THE SIDEBAR AND STYLE THE HEADER
    st.markdown("""
        <style>
            /* Completely hide the sidebar and its toggle button */
            [data-testid="stSidebar"] {display: none;}
            [data-testid="collapsedControl"] {display: none;}
            
            /* Navigation bar styling */
            .nav-container {
                display: flex;
                justify_content: space-between;
                align_items: center;
                padding: 10px 20px;
                background-color: white;
                border-bottom: 1px solid #f0f0f0;
                margin-bottom: 20px;
            }
            .nav-logo {
                font-family: 'Lora', serif;
                font-size: 1.2rem;
                font-weight: bold;
                color: #333;
                text-decoration: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # HORIZONTAL NAVIGATION BAR
    # Using columns to align left (Logo/Home) and right (Resume)
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Logo and Home Page link
        c_logo, c_text = st.columns([1, 15], vertical_alignment="center")
        with c_logo:
            if os.path.exists("assets/logo.png"):
                st.image("assets/logo.png", width=40)
        with c_text:
            st.page_link("Home.py", label="Home Page")
        
    with col2:
        # Resume link
        st.page_link("pages/04_Resume_&_Story.py", label="My Resume & Story")

    st.divider()
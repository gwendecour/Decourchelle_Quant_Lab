import streamlit as st

def render_top_navigation():
    """
    Renders a consistent top navigation bar across all pages.
    """
    # Stylized separator
    st.markdown("---")
    
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        # Link back to Home
        st.page_link("Home.py", label="🏠 Home / Projects Hub", icon="🔙")
        
    with col2:
        st.page_link("pages/04_📜_Resume_&_Story.py", label="My Resume", icon="📄")
        
    with col3:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/ton-profil") # Remplace par ton lien

    st.markdown("---")
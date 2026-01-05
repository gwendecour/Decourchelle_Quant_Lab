import streamlit as st

def render_header():
    """
    Affiche une barre de navigation horizontale et cache la sidebar native.
    """
    # 1. CSS POUR CACHER LA SIDEBAR ET STYLISER LE HEADER
    st.markdown("""
        <style>
            /* Cacher complètement le 'rideau' (sidebar) et le bouton pour l'ouvrir */
            [data-testid="stSidebar"] {display: none;}
            [data-testid="collapsedControl"] {display: none;}
            
            /* Style de la barre de navigation */
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

    # 2. BARRE DE NAVIGATION HORIZONTALE
    # On utilise des colonnes pour aligner gauche (Logo/Home) et droite (CV)
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Lien retour maison (simulé par un bouton lien ou text)
        st.page_link("Home.py", label="🧪 Gwendal's Lab", icon="🏠")
        
    with col2:
        # Lien vers le CV
        st.page_link("pages/04_Resume_&_Story.py", label="My Resume & Story", icon="📜")

    st.divider()
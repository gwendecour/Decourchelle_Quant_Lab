import streamlit as st
from src.shared.ui import render_header

st.set_page_config(page_title="My Resume", layout="wide")
render_header()

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://placehold.co/400x400", caption="Gwendal [Last Name]") # Mets ta photo
    st.download_button(
        label="📄 Download Full Resume (PDF)",
        data=b"Pretend this is a PDF", # Tu pourras charger ton vrai PDF ici
        file_name="Gwendal_Resume.pdf",
        mime="application/pdf",
        use_container_width=True
    )

with col2:
    st.header("My Story")
    st.markdown("""
    I am a **Financial Engineer** passionate about bridging the gap between quantitative finance and modern technology.
    
    My journey started with...
    
    ### Education
    * **Master in Financial Engineering** - [School Name] (202x - 202x)
    
    ### Key Skills
    * **Finance:** Derivatives Pricing, Risk Management, Delta-Hedging.
    * **Tech:** Python (Pandas, Numpy, Scipy), Streamlit, Git, SQL.
    * **Languages:** English (Fluent), French (Native).
    """)
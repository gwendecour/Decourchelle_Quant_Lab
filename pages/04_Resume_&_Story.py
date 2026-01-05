import streamlit as st
from src.shared.ui import render_header

st.set_page_config(page_title="About Me | Resume", layout="wide")
render_header()

# --- Language selector ---
language = st.radio(
    "Language / Langue",
    options=["English 🇬🇧", "Français 🇫🇷"],
    horizontal=True
)

# --- CV download (top, discreet but visible) ---
with open("assets/Gwendal_Decourchelle_Resume.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(
    label="📄 Download Resume (PDF)",
    data=PDFbyte,
    file_name="Gwendal_Decourchelle_Resume.pdf",
    mime="application/pdf",
    use_container_width=False
)

st.divider()

# --- Main content ---
if language == "English 🇬🇧":
    st.header("About Me")

    st.markdown("""
    I am a **French engineering and business school student** with a strong interest in
    **financial markets**, particularly **trading, derivatives, and asset allocation**.

    I am currently pursuing a **double degree** between **École Centrale de Lille**
    and **EDHEC Business School**, where I am specializing in **Financial Engineering**.
    This academic path allows me to combine **quantitative analysis, programming,
    and financial theory**.

    Although I have not yet had the opportunity to gain professional experience in
    **market finance**, I am **highly motivated and proactive**.
    I built this website to **apply what I learn in practice**, develop
    **derivatives pricing and backtesting tools**, and continuously strengthen
    my **coding and financial modeling skills**.

    This project reflects my genuine interest in **market finance** and my ambition
    to build a **solid, hands-on foundation in trading, risk management,
    and portfolio allocation**.
    """)

else:
    st.header("À propos de moi")

    st.markdown("""
    Je suis un **étudiant français en école d’ingénieur et de commerce**,
    passionné par les **marchés financiers**, en particulier le **trading,
    les produits dérivés et l’allocation d’actifs**.

    Je poursuis actuellement un **double diplôme** entre l’**École Centrale de Lille**
    et l’**EDHEC Business School**, avec une spécialisation en
    **ingénierie financière**. Ce parcours me permet de combiner
    **analyse quantitative, programmation et théorie financière**.

    N’ayant pas encore eu l’opportunité d’acquérir une expérience professionnelle
    en **finance de marché**, j’ai choisi d’être **proactif**.
    J’ai créé ce site afin de **mettre en application mes connaissances**,
    développer des **outils de pricing et de backtesting de produits dérivés**,
    et renforcer mes compétences en **programmation et modélisation financière**.

    Ce projet illustre ma **forte motivation** et mon objectif de construire
    une base **solide, concrète et opérationnelle** en **trading,
    gestion du risque et allocation de portefeuille**.
    """)

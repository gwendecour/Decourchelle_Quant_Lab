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
col1, col2 = st.columns(2, gap="large")

if language == "English 🇬🇧":
    with col1:
        st.header("The Engineering & Business Dual Background")
        st.markdown("""
        I am a French student currently pursuing a **double degree** between **École Centrale de Lille** (General Engineering) and **EDHEC Business School** (MSc in Financial Engineering).
        
        This **"double hat"** allows me to bridge the gap between the quantitative rigor of engineering and the economic reality of business. My goal is to deeply understand **market interactions** and grasp the daily reality of market finance.
        
        Although I have not yet had the opportunity to gain professional experience in this field, I chose to be **proactive**. I built this portfolio to prove that I don't just stick to theory: I build, I test, and I solve concrete problems.
        
        My ambition is clear: **to train myself through practice** in order to be fully **operational from day one of my first internship**, regardless of the desk or asset class.
        """)

    with col2:
        st.header("Project Philosophy")
        st.markdown("""
        **An "Open Box" Laboratory**
        
        This website is my personal laboratory. The philosophy is simple: **"Learning by Doing"**. I started with simple concepts (Vanilla Pricing) and progressively moved towards complex ones (Exotics, Dynamic Backtesting) to discover the mechanisms of finance step by step.
        
        **AI as a Learning Accelerator**
        
        I firmly believe that the coding of tomorrow need to be assisted with AI, which is a powerful tool when used correctly. For this project, I used Artificial Intelligence not to write code blindly, but as a **technical mentor** to:
        
        * **Save development time**, allowing me to focus on the financial logic rather than syntax.
        * **Guide my analysis**, suggesting relevant metrics to track and providing ideas on what is pertinent to visualize for a trader.
        * **Deepen my understanding** of different models by challenging my assumptions.
        
        It allowed me to satisfy my curiosity and reach a **level of detail and precision** that I could not have achieved alone in such a short time.
        """)

else:
    with col1:
        st.header("La Double Casquette Ingénieur & Commerce")
        st.markdown("""
        Je suis actuellement un **double cursus** entre l’**École Centrale de Lille** (Ingénieur généraliste) et l’**EDHEC Business School** (MSc in Financial Engineering).
        
        Cette **double casquette** me permet de faire le pont entre la rigueur quantitative de l'ingénieur et la réalité économique de la finance. Mon but est de comprendre les **interactions de marché** et d'appréhender le quotidien de la finance de marché.
        
        N’ayant pas encore d'expérience professionnelle dans ce domaine, j'ai choisi d'être **proactif**. Ce portfolio est ma façon de démontrer que je ne me contente pas de la théorie : je construis, je teste et je cherche à résoudre des problèmes concrets.
        
        Mon ambition est claire : **me former par la pratique** afin d'être **opérationnel dès le premier jour de mon premier stage**, quel que soit le desk ou la classe d'actifs.
        """)

    with col2:
        st.header("Philosophie du Site")
        st.markdown("""
        **Un Laboratoire "Open Box"**
        
        Ce site est mon laboratoire personnel. La philosophie est simple : **"Apprendre en faisant"**. J'ai évolué pas à pas, du simple (Pricing Vanille) au complexe (Produits Exotiques, Backtesting dynamique), pour découvrir les mécanismes de la finance par l'expérience.
        
        **L'IA comme accélérateur de savoir**
        
        Je crois fermement que la développement de demain devra s'aider de l'IA qui est un super outil lorsque bien utilisé. Pour ce projet, j'ai utilisé l'Intelligence Artificielle non pas pour coder à ma place, mais comme un **mentor technique** pour :
        
        * **Gagner du temps sur le développement**, me permettant de me concentrer sur la logique financière plutôt que la syntaxe.
        * **Guider mon analyse**, en me suggérant des pistes sur les métriques pertinentes à afficher et ce qu'il est intéressant de tracer visuellement.
        * **Comprendre les modèles** en profondeur en challengeant mes hypothèses.
        
        Cela m'a permis de répondre à ma curiosité et d'atteindre un **niveau de détail et de précision** que je n'aurais pas pu atteindre tout seul dans ce laps de temps.
        """)
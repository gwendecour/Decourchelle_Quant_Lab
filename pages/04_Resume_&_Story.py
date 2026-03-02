import streamlit as st
from src.shared.ui import render_header

st.set_page_config(page_title="About Me | Resume", page_icon="assets/logo.png", layout="wide")
render_header()

# --- Language selector ---
language = st.radio(
    "Language / Langue",
    options=["English", "Français"],
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

if language == "English":
    with col1:
        st.header("About Me: ")
        st.markdown("""
        I am a French student currently pursuing a dual degree between **École Centrale de Lille** (General Engineering) and **EDHEC Business School** (MSc in Financial Engineering).

        This background allows me to bridge the gap between advanced quantitative rigor and the economic realities of the business. I have a strong, driven interest in **Market Finance**, specifically aiming for roles in **Trading**, **Algorithmic Trading**, or **Quantitative Analysis**. My goal is to deeply understand market dynamics, leverage complex data, and translate mathematical models into actionable strategies.

        While I am at the beginning of my professional journey in this industry, I have chosen to be fiercely proactive. I built this portfolio to demonstrate that I do not just stick to academic theory: I write code, I backtest models, and I solve concrete financial problems.

        My ambition is clear: to train myself through hands-on practice so that I am fully operational from day one of my first internship, whether it involves pricing complex derivatives, developing execution algorithms, or supporting a trading desk.

        Let's connect:

        Email: gwendal.decourchelle@edhec.com

        LinkedIn: www.linkedin.com/in/gwendal-decourchelle

        GitHub: https://github.com/gwendecour
        """)

    with col2:
        st.header("Project Philosophy")
        st.markdown("""
        **An "Open Box" Environment**
        
        This website is a space dedicated to experimentation and discovery. The philosophy is simple: "Learning by Doing". I started with simple concepts (Vanilla Pricing) and progressively moved towards complex ones (Exotics, Dynamic Backtesting) to explore, test, and uncover the mechanisms of quantitative finance step by step.
        
        **AI as a Learning Accelerator**
        
        I firmly believe that the coding of tomorrow need to be assisted with AI, which is a powerful tool when used correctly. For this project, I used Artificial Intelligence not to write code blindly, but as a technical mentor to:
        Save development time, allowing me to focus on the financial logic rather than syntax. 
        Guide my analysis, suggesting relevant metrics to track and providing ideas on what is pertinent to visualize for a trader. 
        Deepen my understanding of different models by challenging my assumptions. 
        It allowed me to satisfy my curiosity and reach a level of detail and precision that I could not have achieved alone in such a short time.
        """)

else:
    with col1:
        st.header("Mon profil")
        st.markdown("""
        Je suis un étudiant français actuellement en double diplôme entre l'École Centrale de Lille (Ingénieur généraliste) et l'EDHEC Business School (MSc en Ingénierie Financière).

        Ce parcours me permet de faire le pont entre une grande rigueur quantitative et les réalités économiques du marché. Je porte un vif intérêt à la finance de marché, avec pour objectif d'évoluer vers des postes en Trading, Trading Algorithmique ou Analyse Quantitative (Quant). Mon but est de comprendre en profondeur les dynamiques de marché, d'exploiter des données complexes et de traduire des modèles mathématiques en stratégies opérationnelles.

        Bien que je sois au début de mon parcours professionnel dans ce secteur, j'ai choisi d'être résolument proactif. J'ai conçu ce portfolio pour démontrer que je ne me limite pas à la théorie académique : je code, je backteste des modèles et je résous des problèmes financiers concrets.

        Mon ambition est claire : me former par la pratique afin d'être pleinement opérationnel dès le premier jour de mon stage, que ce soit pour pricer des produits dérivés complexes, développer des algorithmes d'exécution ou accompagner un desk de trading.

        Contacts :
        Email : gwendal.decourchelle@edhec.com

        LinkedIn : www.linkedin.com/in/gwendal-decourchelle
        
        GitHub : https://github.com/gwendecour""")

    with col2:
        st.header("Philosophie du Site")
        st.markdown("""
        **Un Espace d'Expérimentation "Open Box"**
        
        Ce site est un espace personnel pensé pour expérimenter, tester et découvrir. La philosophie est simple : "Apprendre en faisant". J'ai commencé par des concepts simples (Pricing Vanille) pour évoluer progressivement vers des concepts plus complexes (Exotiques, Backtesting dynamique) afin d'explorer de manière concrète les mécanismes de la finance quantitative étape par étape.
        
        **L'IA comme accélérateur d'apprentissage**
        
        Je crois fermement que le développement de demain devra s'aider de l'IA, qui est un super outil lorsqu'il est bien utilisé. Pour ce projet, j'ai utilisé l'Intelligence Artificielle non pas pour coder à ma place, mais comme un mentor technique pour :
        Gagner du temps sur le développement, me permettant de me concentrer sur la logique financière plutôt que la syntaxe.
        Guider mon analyse, en me suggérant des pistes sur les métriques pertinentes à afficher et ce qu'il est intéressant de tracer visuellement pour un trader.
        Comprendre les modèles en profondeur en challengeant mes hypothèses.
        Cela m'a permis de satisfaire ma curiosité et d'atteindre un niveau de détail et de précision que je n'aurais pas pu atteindre tout seul dans un laps de temps si court.
        """)
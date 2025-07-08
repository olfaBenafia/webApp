import streamlit as st
import base64

# Configuration de la page
st.set_page_config(
    page_title="CARTEX - Analyseur Automatique des Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour le style
st.markdown("""
<style>
    /* Masquer le header et footer par défaut de Streamlit */
    .stApp > header {visibility: hidden;}
    .stApp > footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Style pour le container principal */
    .main-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        min-height: 100vh;
        padding: 20px;
        margin: -1rem;
    }

    /* Style pour la barre de navigation */
    .nav-bar {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 25px;
        padding: 10px 20px;
        margin-bottom: 50px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .nav-item {
        display: inline-block;
        margin: 0 15px;
        padding: 10px 20px;
        color: white;
        text-decoration: none;
        border-radius: 15px;
        transition: all 0.3s ease;
        font-weight: 500;
        cursor: pointer;
    }

    .nav-item:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }

    .nav-item.active {
        background: rgba(255, 255, 255, 0.3);
        color: #fff;
    }

    /* Style pour le logo et titre principal */
    .main-title {
        text-align: center;
        margin: 100px 0;
    }

    .subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 18px;
        font-weight: 300;
        margin-bottom: 30px;
    }

    .cartex-logo {
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff6b6b, #ffd93d, #6bcf7f, #4d9fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
        letter-spacing: 5px;
        margin: 20px 0;
    }

    .divider {
        width: 200px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #fff, transparent);
        margin: 30px auto;
        opacity: 0.5;
    }

    /* Style pour le logo de la société */
    .company-logo {
        position: absolute;
        bottom: 30px;
        right: 30px;
        color: rgba(255, 255, 255, 0.7);
        font-size: 24px;
        font-weight: 300;
        letter-spacing: 2px;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .cartex-logo {
            font-size: 3rem;
        }
        .nav-item {
            margin: 5px;
            padding: 8px 15px;
            font-size: 14px;
        }
        .company-logo {
            bottom: 20px;
            right: 20px;
            font-size: 18px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Container principal
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Barre de navigation
st.markdown("""
<div class="nav-bar">
    <span class="nav-item active">🏠 Accueil</span>
    <span class="nav-item">📋 À propos</span>
    <span class="nav-item">🔍 Scanner</span>
    <span class="nav-item">📊 Tableau de Bord</span>
    <span class="nav-item">📧 Contact</span>
</div>
""", unsafe_allow_html=True)

# Titre principal et logo
st.markdown("""
<div class="main-title animate-fade-in-up">
    <div class="subtitle">Analyseur Automatique des Documents</div>
    <div class="cartex-logo">CARTEX</div>
    <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# Logo de la société
st.markdown("""
<div class="company-logo">
    Amaris Consulting
</div>
""", unsafe_allow_html=True)

# Fermeture du container principal
st.markdown('</div>', unsafe_allow_html=True)

# Gestion de la navigation avec session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'accueil'

# JavaScript pour la navigation (optionnel pour les interactions futures)
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
});
</script>
""", unsafe_allow_html=True)

# Informations pour le développement
st.markdown("""
<div style="position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px; opacity: 0.7;">
    Interface d'accueil CARTEX - Prête pour navigation
</div>
""", unsafe_allow_html=True)
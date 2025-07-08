import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard CIN/Carte Grise", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("static/bi/resultats_dashboard.csv")

df = load_data()

st.title("📊 Dashboard des Données CIN & Carte Grise")

# Filtres
col1, col2 = st.columns(2)
with col1:
    sexe = st.multiselect("Sexe", options=df['sexe'].unique(), default=df['sexe'].unique())
with col2:
    wilaya = st.multiselect("Wilaya (lieu de naissance)", options=df['wilaya'].unique(), default=df['wilaya'].unique())

# Application des filtres
df_filtre = df[(df['sexe'].isin(sexe)) & (df['wilaya'].isin(wilaya))]

# KPIs
st.markdown("### 📌 Statistiques")
col1, col2, col3 = st.columns(3)
col1.metric("Nombre total", len(df_filtre))
col2.metric("Hommes", len(df_filtre[df_filtre['sexe'] == 'Homme']))
col3.metric("Femmes", len(df_filtre[df_filtre['sexe'] == 'Femme']))

# Graphiques
st.markdown("### 📈 Répartition par Wilaya")
st.bar_chart(df_filtre['wilaya'].value_counts())

st.markdown("### 📅 Répartition par Jour")
st.line_chart(df_filtre['jour'].value_counts().sort_index())

# Table
st.markdown("### 🧾 Données filtrées")
st.dataframe(df_filtre)

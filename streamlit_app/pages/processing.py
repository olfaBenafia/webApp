import streamlit as st

def display_key_value(title, data_dict):
    st.subheader(title)
    if not data_dict:
        st.write("Aucune donnée disponible.")
        return
    for key, value in data_dict.items():
        st.markdown(f"**{key}**: {value}")

def app():
    st.title("Résultats de l'analyse")

    result = st.session_state.get('ocr_result')
    if not result:
        st.warning("Aucun résultat disponible. Veuillez revenir et analyser un document.")
        if st.button("← Retour à l'upload"):
            st.session_state.current_page = 'image_upload'
            st.experimental_rerun()
        return

    st.header("Données Recto OCR brutes")
    display_key_value("Recto OCR brut", result.get('data_recto'))

    st.header("Données Verso OCR brutes")
    display_key_value("Verso OCR brut", result.get('data_verso'))

    st.header("Données Recto Traduites")
    display_key_value("Recto traduit", result.get('data_recto_fr'))

    st.header("Données Verso Traduites")
    display_key_value("Verso traduit", result.get('data_verso_fr'))

    if st.button("← Retour à l'upload"):
        st.session_state.current_page = 'image_upload'
        st.experimental_rerun()

# utils.py

import streamlit as st

def navigate_to(page_name: str):
    st.session_state.current_page = page_name
    st.experimental_rerun()

def initialize_session():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'document_selection'
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'upload_progress' not in st.session_state:
        st.session_state.upload_progress = 0

def show_progress(progress_text="Traitement en cours..."):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f'{progress_text} {i+1}%')

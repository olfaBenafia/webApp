import streamlit as st
import requests
import json
from PIL import Image
import io

# Configuration
FLASK_API_BASE_URL = "http://localhost:5000"  # Adjust this to match your Flask server


def image_upload_page():
    """Scanner page with document type selection and dual upload"""
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        min-height: 100vh;
    }

    .scanner-container {
        background: rgba(30, 60, 114, 0.3);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem auto;
        max-width: 800px;
    }

    .page-title {
        color: white;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .section-title {
        color: white;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: rgba(0, 188, 212, 0.5);
        background: rgba(255, 255, 255, 0.1);
    }

    .upload-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    .upload-limit {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(45deg, #00bcd4, #0097a7) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 188, 212, 0.3) !important;
        margin: 0.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 188, 212, 0.4) !important;
    }

    .method-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 2rem 0;
    }

    .method-button {
        background: rgba(0, 188, 212, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(0, 188, 212, 0.5) !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        min-width: 120px !important;
    }

    .method-button:hover {
        background: rgba(0, 188, 212, 1) !important;
        transform: translateY(-1px) !important;
    }

    .method-button.selected {
        background: #00bcd4 !important;
        border-color: #00bcd4 !important;
        box-shadow: 0 4px 15px rgba(0, 188, 212, 0.4) !important;
    }

    /* Navigation styles */
    .nav-container {
        background: rgba(30, 60, 114, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .stRadio > div > label {
        color: white !important;
        font-size: 1.1rem !important;
    }

    .stRadio > div > label > div {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
    }

    .stFileUploader > div:hover {
        border-color: rgba(0, 188, 212, 0.5) !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    .stFileUploader label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1.1rem !important;
    }

    .stFileUploader small {
        color: rgba(255, 255, 255, 0.6) !important;
    }

    /* Alert styles */
    .alert-success {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .alert-error {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        color: #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation
    render_navigation('scanner')

    # Main content
    st.markdown('<div class="scanner-container">', unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="page-title">Type du Document</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'selected_document_type' not in st.session_state:
        st.session_state.selected_document_type = None
    if 'selected_method' not in st.session_state:
        st.session_state.selected_method = None
    if 'processing_result' not in st.session_state:
        st.session_state.processing_result = None

    # Document type selection
    document_types = [
        "Carte d'identite Nationale",
        "Passeport",
        "Carte Grise",
        "Permis"
    ]

    selected_doc = st.radio(
        "",
        document_types,
        key="document_type_radio"
    )

    st.session_state.selected_document_type = selected_doc

    # Method selection buttons
    st.markdown('<div class="method-buttons">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Image", key="image_method"):
            st.session_state.selected_method = 'image'
            st.rerun()

    with col2:
        if st.button("Camera", key="camera_method"):
            st.session_state.selected_method = 'camera'
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Show upload sections if method is selected
    if st.session_state.selected_method:
        st.markdown(
            f'<div class="section-title">Méthode sélectionnée: {st.session_state.selected_method.title()}</div>',
            unsafe_allow_html=True)

        if selected_doc == "Carte d'identite Nationale":
            # Front side upload
            st.markdown('<div class="section-title">Importez la Face Avant</div>', unsafe_allow_html=True)

            front_file = st.file_uploader(
                "Drag and drop file here",
                type=['jpg', 'jpeg', 'png'],
                key="front_upload",
                help="Limit 200MB per file • PNG, JPG, JPEG"
            )

            if front_file:
                st.image(front_file, caption="Face Avant", use_column_width=True)

            # Back side upload
            st.markdown('<div class="section-title">Importez la Face Arrière</div>', unsafe_allow_html=True)

            back_file = st.file_uploader(
                "Drag and drop file here",
                type=['jpg', 'jpeg', 'png'],
                key="back_upload",
                help="Limit 200MB per file • PNG, JPG, JPEG"
            )

            if back_file:
                st.image(back_file, caption="Face Arrière", use_column_width=True)

            # Submit button
            if front_file and back_file:
                if st.button("Soumettre", key="submit_btn"):
                    with st.spinner("Traitement en cours..."):
                        result = process_cin_documents(front_file, back_file)
                        st.session_state.processing_result = result
                        if result and 'error' not in result:
                            st.success("✅ Traitement terminé avec succès!")
                            display_cin_results(result)
                        else:
                            st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")

        elif selected_doc == "Carte Grise":
            # Front side upload
            st.markdown('<div class="section-title">Importez la Face Avant</div>', unsafe_allow_html=True)

            front_file = st.file_uploader(
                "Drag and drop file here",
                type=['jpg', 'jpeg', 'png'],
                key="cg_front_upload",
                help="Limit 200MB per file • PNG, JPG, JPEG"
            )

            if front_file:
                st.image(front_file, caption="Face Avant", use_column_width=True)

            # Back side upload
            st.markdown('<div class="section-title">Importez la Face Arrière</div>', unsafe_allow_html=True)

            back_file = st.file_uploader(
                "Drag and drop file here",
                type=['jpg', 'jpeg', 'png'],
                key="cg_back_upload",
                help="Limit 200MB per file • PNG, JPG, JPEG"
            )

            if back_file:
                st.image(back_file, caption="Face Arrière", use_column_width=True)

            # Submit button
            if front_file and back_file:
                if st.button("Soumettre", key="submit_cg_btn"):
                    with st.spinner("Traitement en cours..."):
                        result = process_carte_grise_documents(front_file, back_file)
                        st.session_state.processing_result = result
                        if result and 'error' not in result:
                            st.success("✅ Traitement terminé avec succès!")
                            display_carte_grise_results(result)
                        else:
                            st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")

        else:
            # Single document upload for other types
            st.markdown(f'<div class="section-title">Importez le document: {selected_doc}</div>',
                        unsafe_allow_html=True)

            document_file = st.file_uploader(
                "Drag and drop file here",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                key="document_upload",
                help="Limit 200MB per file • PNG, JPG, JPEG, PDF"
            )

            if document_file:
                if document_file.type.startswith('image'):
                    st.image(document_file, caption=f"{selected_doc}", use_column_width=True)
                else:
                    st.success(f"Fichier PDF téléchargé: {document_file.name}")

                # Submit button
                if st.button("Soumettre", key="submit_single_btn"):
                    st.info("Traitement pour ce type de document sera implémenté prochainement.")

    st.markdown('</div>', unsafe_allow_html=True)


def process_cin_documents(front_file, back_file):
    """Process CIN documents by calling Flask API"""
    try:
        # Prepare files for upload
        files = {
            'recto': (front_file.name, front_file.getvalue(), front_file.type),
            'verso': (back_file.name, back_file.getvalue(), back_file.type)
        }

        # Make API call
        response = requests.post(
            f"{FLASK_API_BASE_URL}/upload_cin",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}

    except requests.exceptions.ConnectionError:
        return {
            "error": "Impossible de se connecter au serveur. Assurez-vous que l'API Flask est en cours d'exécution."}
    except requests.exceptions.Timeout:
        return {"error": "Timeout - Le traitement a pris trop de temps."}
    except Exception as e:
        return {"error": f"Erreur lors du traitement: {str(e)}"}


def process_carte_grise_documents(front_file, back_file):
    """Process Carte Grise documents by calling Flask API"""
    try:
        # Prepare files for upload
        files = {
            'recto': (front_file.name, front_file.getvalue(), front_file.type),
            'verso': (back_file.name, back_file.getvalue(), back_file.type)
        }

        # Make API call
        response = requests.post(
            f"{FLASK_API_BASE_URL}/upload_cartegrise",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}

    except requests.exceptions.ConnectionError:
        return {
            "error": "Impossible de se connecter au serveur. Assurez-vous que l'API Flask est en cours d'exécution."}
    except requests.exceptions.Timeout:
        return {"error": "Timeout - Le traitement a pris trop de temps."}
    except Exception as e:
        return {"error": f"Erreur lors du traitement: {str(e)}"}


def display_cin_results(result):
    """Display CIN processing results"""
    st.markdown("### Résultats du traitement")

    # Display in expandable sections
    with st.expander("📄 Données Face Avant (Original)"):
        if 'data_recto' in result:
            st.json(result['data_recto'])
        else:
            st.write("Aucune donnée disponible")

    with st.expander("📄 Données Face Arrière (Original)"):
        if 'data_verso' in result:
            st.json(result['data_verso'])
        else:
            st.write("Aucune donnée disponible")

    with st.expander("🇫🇷 Données Face Avant (Traduites)"):
        if 'data_recto_fr' in result:
            st.json(result['data_recto_fr'])
        else:
            st.write("Aucune donnée disponible")

    with st.expander("🇫🇷 Données Face Arrière (Traduites)"):
        if 'data_verso_fr' in result:
            st.json(result['data_verso_fr'])
        else:
            st.write("Aucune donnée disponible")


def display_carte_grise_results(result):
    """Display Carte Grise processing results"""
    st.markdown("### Résultats du traitement")

    # Display in expandable sections
    with st.expander("📄 Données Face Avant"):
        if 'data_recto' in result:
            st.json(result['data_recto'])
        else:
            st.write("Aucune donnée disponible")

    with st.expander("📄 Données Face Arrière"):
        if 'data_verso' in result:
            st.json(result['data_verso'])
        else:
            st.write("Aucune donnée disponible")


def render_navigation(active_page):
    """Render navigation bar"""
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🏠 Accueil", key="nav_home", disabled=(active_page == 'home')):
            st.session_state.current_page = 'landing'
            st.rerun()

    with col2:
        if st.button("📄 À propos", key="nav_about", disabled=(active_page == 'about')):
            st.session_state.current_page = 'about'
            st.rerun()

    with col3:
        if st.button("📱 Scanner", key="nav_scanner", disabled=(active_page == 'scanner')):
            st.session_state.current_page = 'scanner'
            st.rerun()

    with col4:
        if st.button("📊 Tableau de Bord", key="nav_dashboard", disabled=(active_page == 'dashboard')):
            st.session_state.current_page = 'dashboard'
            st.rerun()

    with col5:
        if st.button("📧 Contact", key="nav_contact", disabled=(active_page == 'contact')):
            st.session_state.current_page = 'contact'
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# Processing page - now integrated into main page
def processing_page():
    """Processing page - this is now handled inline"""
    st.markdown("Processing is now handled inline in the main page.")
    if st.button("← Retour au Scanner"):
        st.session_state.current_page = 'scanner'
        st.rerun()


# Main function to run the app
def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'scanner'

    # Route to appropriate page
    if st.session_state.current_page == 'scanner':
        image_upload_page()
    elif st.session_state.current_page == 'processing':
        processing_page()
    # Add other pages as needed


if __name__ == "__main__":
    main()
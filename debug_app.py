import streamlit as st
import requests
import json
import time
from PIL import Image
import io

# Configuration
FLASK_API_BASE_URL = "http://localhost:5000"  # Adjust this to match your Flask server


def test_flask_connection():
    """Test if Flask server is running"""
    try:
        response = requests.get(f"{FLASK_API_BASE_URL}/info", timeout=5)
        if response.status_code == 200:
            return True, "✅ Flask server is running"
        else:
            return False, f"❌ Flask server responded with status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to Flask server. Make sure it's running on http://localhost:5000"
    except Exception as e:
        return False, f"❌ Error connecting to Flask server: {str(e)}"


def process_cin_documents_debug(front_file, back_file):
    """Process CIN documents with debug information"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Check Flask connection
        status_text.text("🔍 Checking Flask server connection...")
        progress_bar.progress(10)

        is_connected, message = test_flask_connection()
        if not is_connected:
            return {"error": message}

        st.success(message)

        # Step 2: Prepare files
        status_text.text("📁 Preparing files for upload...")
        progress_bar.progress(20)

        files = {
            'recto': (front_file.name, front_file.getvalue(), front_file.type),
            'verso': (back_file.name, back_file.getvalue(), back_file.type)
        }

        # Show file info
        st.info(f"Front file: {front_file.name} ({len(front_file.getvalue())} bytes)")
        st.info(f"Back file: {back_file.name} ({len(back_file.getvalue())} bytes)")

        # Step 3: Make API call
        status_text.text("🚀 Sending files to Flask server...")
        progress_bar.progress(30)

        start_time = time.time()

        # Make API call with progress updates
        response = requests.post(
            f"{FLASK_API_BASE_URL}/upload_cin",
            files=files,
            timeout=300  # 5 minutes timeout
        )

        end_time = time.time()
        processing_time = end_time - start_time

        progress_bar.progress(100)
        status_text.text(f"✅ Processing completed in {processing_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Success! Processing took {processing_time:.2f} seconds")
            return result
        else:
            error_msg = f"API Error: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('error', response.text)}"
            except:
                error_msg += f" - {response.text}"
            return {"error": error_msg}

    except requests.exceptions.Timeout:
        return {"error": "⏱️ Timeout - Le traitement a pris trop de temps (plus de 5 minutes)"}
    except requests.exceptions.ConnectionError:
        return {"error": "🔌 Connection Error - Impossible de se connecter au serveur Flask"}
    except Exception as e:
        return {"error": f"❌ Unexpected error: {str(e)}"}


def debug_page():
    """Debug page for testing"""
    st.title("🔧 Debug Page")

    # Test Flask connection
    if st.button("Test Flask Connection"):
        with st.spinner("Testing connection..."):
            is_connected, message = test_flask_connection()
            if is_connected:
                st.success(message)
            else:
                st.error(message)

    # File upload for testing
    st.subheader("Test File Upload")

    front_file = st.file_uploader("Front Image", type=['jpg', 'jpeg', 'png'])
    back_file = st.file_uploader("Back Image", type=['jpg', 'jpeg', 'png'])

    if front_file and back_file:
        col1, col2 = st.columns(2)

        with col1:
            st.image(front_file, caption="Front", use_column_width=True)
            st.text(f"Size: {len(front_file.getvalue())} bytes")
            st.text(f"Type: {front_file.type}")

        with col2:
            st.image(back_file, caption="Back", use_column_width=True)
            st.text(f"Size: {len(back_file.getvalue())} bytes")
            st.text(f"Type: {back_file.type}")

        if st.button("🚀 Process with Debug"):
            result = process_cin_documents_debug(front_file, back_file)

            if 'error' in result:
                st.error(result['error'])
            else:
                st.success("Processing completed successfully!")

                # Show results
                with st.expander("📊 Full Results"):
                    st.json(result)

                # Show key data
                if 'data_recto_fr' in result:
                    with st.expander("🇫🇷 French Data - Front"):
                        st.json(result['data_recto_fr'])

                if 'data_verso_fr' in result:
                    with st.expander("🇫🇷 French Data - Back"):
                        st.json(result['data_verso_fr'])


def main():
    st.set_page_config(page_title="Debug App", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose page", ["Debug", "Original App"])

    if page == "Debug":
        debug_page()
    else:
        st.title("Original App")
        st.info("Use the debug page to test your Flask connection and file processing")


if __name__ == "__main__":
    main()
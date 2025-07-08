import datetime
import paddle

paddle.utils.run_check()
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from werkzeug.utils import secure_filename
import os
from DetectCIN import Decoupage_information, compare_ocrs
from DetectCarte_Grise import compare_ocrs_CG
from Traduction import Traductionfinale, inserer_BD
from Traduction import Traductionfinale, inserer_BD, save_dashboard_data


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Increase upload size limit
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_PATH, exist_ok=True)


@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Document Processing API',
        'version': '1.0',
        'endpoints': {
            'cin': '/upload_cin',
            'carte_grise': '/upload_cartegrise',
            'info': '/info'
        }
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Info endpoint"""
    return jsonify({
        'message': 'Document Processing API',
        'supported_documents': ['CIN', 'Carte Grise'],
        'methods': ['POST'],
        'required_files': ['recto', 'verso']
    }), 200


@app.route('/upload_cin', methods=['POST'])
def upload_cin():
    """Handle CIN document upload and processing"""
    try:
        # Check if files are present
        if 'recto' not in request.files or 'verso' not in request.files:
            return jsonify({'error': 'Both recto and verso files are required'}), 400

        upload_file_recto = request.files['recto']
        upload_file_verso = request.files['verso']

        # Check if files are actually selected
        if upload_file_recto.filename == '' or upload_file_verso.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        upload_time = datetime.datetime.now()

        # Save files
        filename_recto = secure_filename(upload_file_recto.filename)
        path_save_recto = os.path.join(UPLOAD_PATH, f"recto_{upload_time.strftime('%Y%m%d_%H%M%S')}_{filename_recto}")
        upload_file_recto.save(path_save_recto)

        filename_verso = secure_filename(upload_file_verso.filename)
        path_save_verso = os.path.join(UPLOAD_PATH, f"verso_{upload_time.strftime('%Y%m%d_%H%M%S')}_{filename_verso}")
        upload_file_verso.save(path_save_verso)

        # Process recto
        print(f"Processing recto: {path_save_recto}")
        d = compare_ocrs(path_save_recto, 'recto')
        if d is None:
            return jsonify({'error': 'OCR processing failed for recto image'}), 500

        d = Decoupage_information(d)
        if d is None:
            return jsonify({'error': 'Information extraction failed for recto'}), 500

        dict_final_recto = Traductionfinale(d, 'recto')
        if dict_final_recto is None:
            return jsonify({'error': 'Translation failed for recto'}), 500

        # Process verso
        print(f"Processing verso: {path_save_verso}")
        d_verso = compare_ocrs(path_save_verso, 'verso')
        if d_verso is None:
            return jsonify({'error': 'OCR processing failed for verso image'}), 500

        d_final_verso = Traductionfinale(d_verso, 'verso')
        if d_final_verso is None:
            return jsonify({'error': 'Translation failed for verso'}), 500

        # Insert into DB
        try:
            inserer_BD(dict_final_recto, upload_time, 'recto')
            inserer_BD(d_final_verso, upload_time, 'verso')
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            # Continue processing even if DB insert fails
            pass

        # Clean up files (optional)
        try:
            os.remove(path_save_recto)
            os.remove(path_save_verso)
        except:
            pass

        # Return JSON result
        return jsonify({
            'success': True,
            'message': 'CIN processing completed successfully',
            'upload_image_recto': filename_recto,
            'upload_image_verso': filename_verso,
            'data_recto': d,
            'data_verso': d_verso,
            'data_recto_fr': dict_final_recto,
            'data_verso_fr': d_final_verso,
            'timestamp': upload_time.isoformat()
        }), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in upload_cin: {str(e)}")
        print(f"Traceback: {error_trace}")

        return jsonify({
            'error': str(e),
            'message': 'An error occurred during CIN processing',
            'traceback': error_trace
        }), 500


@app.route('/upload_cartegrise', methods=['POST'])
def upload_cartegrise():
    """Handle Carte Grise document upload and processing"""
    try:
        if 'recto' not in request.files or 'verso' not in request.files:
            return jsonify({'error': 'Both recto and verso files are required'}), 400

        upload_file_recto = request.files['recto']
        upload_file_verso = request.files['verso']

        # Check if files are actually selected
        if upload_file_recto.filename == '' or upload_file_verso.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        upload_time = datetime.datetime.now()

        # Save files
        filename_recto = secure_filename(upload_file_recto.filename)
        path_save_recto = os.path.join(UPLOAD_PATH,
                                       f"cg_recto_{upload_time.strftime('%Y%m%d_%H%M%S')}_{filename_recto}")
        upload_file_recto.save(path_save_recto)

        filename_verso = secure_filename(upload_file_verso.filename)
        path_save_verso = os.path.join(UPLOAD_PATH,
                                       f"cg_verso_{upload_time.strftime('%Y%m%d_%H%M%S')}_{filename_verso}")
        upload_file_verso.save(path_save_verso)

        # Process files
        print(f"Processing Carte Grise recto: {path_save_recto}")
        dict_final_recto = compare_ocrs_CG(path_save_recto, 'recto')

        print(f"Processing Carte Grise verso: {path_save_verso}")
        d_verso = compare_ocrs_CG(path_save_verso, 'verso')

        # Clean up files (optional)
        try:
            os.remove(path_save_recto)
            os.remove(path_save_verso)
        except:
            pass

        return jsonify({
            'success': True,
            'message': 'Carte Grise processing completed successfully',
            'upload_image_recto': filename_recto,
            'upload_image_verso': filename_verso,
            'data_recto': dict_final_recto,
            'data_verso': d_verso,
            'timestamp': upload_time.isoformat()
        }), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in upload_cartegrise: {str(e)}")
        print(f"Traceback: {error_trace}")

        return jsonify({
            'error': str(e),
            'message': 'An error occurred during Carte Grise processing',
            'traceback': error_trace
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/upload_cin', '/upload_cartegrise', '/info']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred on the server'
    }), 500


if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Upload directory: {UPLOAD_PATH}")
    print("Available endpoints:")
    print("  - POST /upload_cin")
    print("  - POST /upload_cartegrise")
    print("  - GET /info")

    app.run(debug=True, host='0.0.0.0', port=5000)
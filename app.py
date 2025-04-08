from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import pandas as pd
import os
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
from utils.face_recognition_utils import FaceRecognitionSystem
from PIL import Image
import requests
import io
import base64
import time
from utils.db_utils import init_db, add_user, get_user, verify_user, check_user_exists, list_all_users
import sqlite3
import logging
import face_recognition

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure random key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Session timeout

# Ensure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Create users.xlsx if it doesn't exist
if not os.path.exists('data/users.xlsx'):
    df = pd.DataFrame(columns=['username', 'email', 'password', 'phone', 'aadhar'])
    df.to_excel('data/users.xlsx', index=False)

# Create face_data.xlsx if it doesn't exist
if not os.path.exists('data/face_data.xlsx'):
    face_df = pd.DataFrame(columns=['name', 'age', 'gender', 'phone', 'aadhar', 'image_filename'])
    face_df.to_excel('data/face_data.xlsx', index=False)

# Create faces directory if it doesn't exist
if not os.path.exists('data/faces'):
    os.makedirs('data/faces')

# Initialize SocketIO and FaceRecognitionSystem
socketio = SocketIO(app)
face_system = FaceRecognitionSystem()

# Add this to ensure the session works properly
@app.before_request
def before_request():
    session.permanent = True

# Initialize database at startup
with app.app_context():
    try:
        init_db()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Log received credentials (remove in production)
        app.logger.info(f"Login attempt for username: {username}")
        
        # Check if user exists first
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT username FROM users WHERE username = ?', (username,))
        user_exists = c.fetchone()
        conn.close()
        
        if not user_exists:
            return render_template('login.html', 
                error="Username not found. Please sign up first.")
        
        if verify_user(username, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', 
                error="Invalid username or password.")
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not all([username, email, password, confirm_password]):
            return render_template('signup.html', error="All fields are required")
        
        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")
        
        # Try to add user to database
        success, message = add_user(username, email, password)
        
        if success:
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            return render_template('signup.html', error=message)
    
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session.get('username'))

@app.route('/streaming')
def streaming():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('streaming.html')

@app.route('/recorded_data')
def recorded_data():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('recorded_data.html')

@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400

        file = request.files['face_image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Extract additional form data
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        aadhar = request.form.get('aadhar')

        # Read the image file
        image = face_recognition.load_image_file(file)
        success = face_system.add_new_face(image, {
            'name': name,
            'age': age,
            'gender': gender,
            'phone': phone,
            'aadhar': aadhar
        })

        if success:
            return jsonify({'success': True, 'message': 'Face registered successfully'})
        else:
            return jsonify({'success': False, 'error': 'No face detected in the image'}), 400

    return render_template('load_data.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Add this new route for video feed
@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image
        encoded_data = data.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        processed_frame, recognition_results = face_system.process_frame(frame)
        
        if recognition_results:
            # Emit recognition results
            socketio.emit('recognition_result', recognition_results)
            
            # Convert processed frame back to base64
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('processed_frame', f'data:image/jpeg;base64,{processed_frame_b64}')
    except Exception as e:
        print(f"Error processing frame: {e}")

def generate_ai_welcome_image():
    """
    Generate or retrieve an AI-based welcome image
    For now, we'll use a placeholder image, but you can integrate with AI image generation services
    """
    try:
        # Replace this URL with your AI image generation service
        response = requests.get('https://api.placeholder.com/ai-face/800x600')
        img = Image.open(io.BytesIO(response.content))
        img_path = 'static/images/generated/welcome.png'
        img.save(img_path)
        return img_path
    except Exception as e:
        print(f"Error generating AI image: {e}")
        return 'static/images/default-welcome.png'

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/check_users')
def check_users():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT username, email FROM users')
        users = c.fetchall()
        conn.close()
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/test_db')
def test_db():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # Check if users table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        tables = c.fetchall()
        
        # Get all users
        c.execute("SELECT username, email FROM users")
        users = c.fetchall()
        
        conn.close()
        
        return {
            'tables': tables,
            'users': users
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/debug_users')
def debug_users():
    """Temporary route to check registered users"""
    users = list_all_users()
    return {'registered_users': users}

@app.route('/reset_password', methods=['POST'])
def reset_password():
    username = request.form.get('username')
    new_password = request.form.get('new_password')
    
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # Update password
        hashed_password = generate_password_hash(new_password)
        c.execute('UPDATE users SET password = ? WHERE username = ?', 
                 (hashed_password, username))
        
        if c.rowcount > 0:
            conn.commit()
            return {'success': True, 'message': 'Password reset successful'}
        else:
            return {'success': False, 'message': 'User not found'}
            
    except Exception as e:
        return {'success': False, 'message': str(e)}
    finally:
        conn.close()

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    image = face_recognition.load_image_file(file)
    frame, recognition_data = face_system.process_frame(image)

    if recognition_data:
        return jsonify(recognition_data)
    else:
        return jsonify({'error': 'No face detected'}), 400

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video file temporarily
    temp_path = 'temp_video.mp4'
    file.save(temp_path)

    try:
        # Open the video file
        video = cv2.VideoCapture(temp_path)
        
        # Process first frame only for now (can be extended to process multiple frames)
        ret, frame = video.read()
        if not ret:
            return jsonify({'error': 'Could not read video file'}), 400

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame using face recognition system
        frame, recognition_data = face_system.process_frame(rgb_frame)

        # Clean up
        video.release()
        os.remove(temp_path)

        if recognition_data:
            return jsonify(recognition_data)
        else:
            return jsonify({'error': 'No face detected in video'}), 400

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\nServer is running at: http://127.0.0.1:5000\n")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True) 
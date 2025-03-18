from flask import Flask, render_template, request, session, redirect, url_for, flash
import pyrebase
import numpy as np
import json
import base64
import csv
import requests
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.src import image
from keras.src import load_model

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Consider using environment variables for this

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup, not on every request
try:
    my_model = load_model('model_trained.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    my_model = None

# Firebase configuration - move to environment variables in production
config = {
    'apiKey': "add-key",  # Use environment variables
    'authDomain': "",
    'projectId': "",
    'storageBucket': "",
    'messagingSenderId': "",
    'appId': "",
    'measurementId': "",
    'databaseURL': ""
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# API key for nutrition data - use environment variables
NUTRITION_API_KEY = 'i/k4999mImdYl93Oag/aEQ==lt5yHMVADykrrlQ1'  # Move to environment variable

def load_food_names_from_csv(csv_file):
    """Load food names from CSV file"""
    food_names = []
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                food_names.extend(row)
        return food_names
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return []

# Load food names at startup
food_list = load_food_names_from_csv("Food_List.csv")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_top_classes(model, image_path, top=3):
    """Predict top food classes from an image"""
    try:
        img = image.load_img(image_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        top_indices = pred.argsort()[0][-top:][::-1]
        
        # Check if indices are valid
        if max(top_indices) >= len(food_list):
            return [("Error: Model prediction index out of range", 0)]
            
        top_predictions = [(food_list[i], float(pred[0][i])) for i in top_indices]
        return top_predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return [("Error during prediction", 0)]

def get_nutrient_info(query):
    """Get nutrition information from API"""
    api_url = f'https://api.api-ninjas.com/v1/nutrition?query={query}'
    try:
        response = requests.get(api_url, headers={'X-Api-Key': NUTRITION_API_KEY}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return [{"name": "Error", "message": f"API returned status code {response.status_code}"}]
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return [{"name": "Error", "message": "Failed to connect to nutrition API"}]

@app.route('/', methods=['POST', 'GET'])
def index():
    """Login page route"""
    if 'user' in session:
        return redirect(url_for('upload_file'))
        
    if request.method == "POST":
        email = request.form.get('username')
        password = request.form.get('password')
        
        if not email or not password:
            flash("Email and password are required")
            return render_template('home.html')
            
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = email
            return redirect(url_for('upload_file'))
        except Exception as e:
            flash("Failed to login. Please check your credentials.")
            return render_template('home.html')
            
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page route"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash("Email and password are required")
            return render_template('signup.html')
            
        try:
            user = auth.create_user_with_email_and_password(email, password)
            flash("Account created successfully! Please login.")
            return redirect(url_for('index'))
        except Exception as e:
            flash("Failed to create account. Email may already be in use.")
            return render_template('signup.html')
            
    return render_template('signup.html')

@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    """Password reset page route"""
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash("Email is required")
            return render_template('forget_password.html')
            
        try:
            auth.send_password_reset_email(email)
            flash("Password reset email sent. Please check your inbox.")
            return redirect(url_for('index'))
        except Exception as e:
            flash("Failed to send reset email. Please check if the email is registered.")
            return render_template('forget_password.html')
            
    return render_template('forget_password.html')

@app.route('/logout')
def logout():
    """Logout route"""
    session.pop('user', None)
    flash("You have been logged out")
    return redirect(url_for('index'))

@app.route('/upload')
def upload_file():
    """Upload page route"""
    if 'user' not in session:
        flash("Please login first")
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict food from uploaded image"""
    if 'user' not in session:
        flash("Please login first")
        return redirect(url_for('index'))
        
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('upload_file'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('upload_file'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        session['uploaded_filepath'] = filepath
        
        if my_model is None:
            flash("Model not loaded properly")
            return redirect(url_for('upload_file'))
            
        top_predictions = predict_top_classes(my_model, filepath)
        
        # Encode image for display
        with open(filepath, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        return render_template('result.html', 
                              top_predictions=top_predictions, 
                              image=encoded_image)
    else:
        flash(f"Invalid file type. Please upload {', '.join(ALLOWED_EXTENSIONS)}")
        return redirect(url_for('upload_file'))

@app.route('/verify_result', methods=['POST'])
def verify_result():
    """Verify selected food prediction"""
    if 'user' not in session:
        return redirect(url_for('index'))
        
    selected_food = request.form.get('food_prediction')
    if not selected_food:
        flash("No food selected")
        return redirect(url_for('upload_file'))
        
    uploaded_filepath = session.get('uploaded_filepath')
    if not uploaded_filepath or not os.path.exists(uploaded_filepath):
        flash("Image not found")
        return redirect(url_for('upload_file'))
        
    # Encode image for display
    with open(uploaded_filepath, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        
    return render_template('result.html', 
                          selected_food=selected_food, 
                          image=encoded_image)

@app.route('/get_nutrient', methods=['POST'])
def display_nutrient():
    """Display nutrient information for selected food"""
    if 'user' not in session:
        return redirect(url_for('index'))
        
    selected_food = request.form.get('confirmedValue')
    food_weight = request.form.get('weight', '100')
    
    if not selected_food:
        flash("No food selected")
        return redirect(url_for('upload_file'))
        
    query = f"{food_weight} gram of {selected_food}"
    nutrient_data = get_nutrient_info(query)
    
    if not nutrient_data:
        flash("No nutrition data found")
        return redirect(url_for('upload_file'))
        
    return render_template('nutrition.html', nutrient=nutrient_data[0])

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# Clean up uploaded files periodically (would be better with a scheduled task)
@app.before_request
def cleanup_old_files():
    """Clean up old uploaded files"""
    # This is a simple implementation; consider using a proper scheduled task manager
    import time
    cutoff = time.time() - 3600  # Files older than 1 hour
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing {filepath}: {e}")

if __name__ == "__main__":
    app.run(debug=False, port=8080)
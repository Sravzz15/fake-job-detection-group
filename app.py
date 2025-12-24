import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
import joblib
import os
import sqlite3
import subprocess 
from datetime import datetime
import io
import json

# ==========================================
# CONFIGURATION
# ==========================================
app = Flask(__name__)
app.secret_key = "mysecretkey123" 

# Dynamic Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'fake_job_model.pkl')
VECTORIZER_FILE = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
DB_PATH = os.path.join(BASE_DIR, "job_predictions.db")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Load Model
def load_model_resources():
    global model, vectorizer
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
    except:
        print("Model files not found. Please train model first.")

load_model_resources()

# ==========================================
# DATABASE HELPER FUNCTIONS
# ==========================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    return conn

# ==========================================
# DATABASE HELPER FUNCTIONS
# ==========================================

def log_to_db(description, prediction, confidence):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO predictions (job_description, prediction, confidence, flagged, timestamp) VALUES (?, ?, ?, ?, ?)",
            (description, prediction, confidence, 'No', current_time)
        )
        post_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return post_id
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def log_retrain_event(accuracy, record_count, source="Default Dataset"):
    """Task 2: Save metadata with CORRECT LOCAL TIME"""
    try:
        conn = get_db_connection()
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrain_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                accuracy REAL,
                record_count INTEGER,
                training_source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # --- TIMEZONE FIX: Get Local System Time ---
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            "INSERT INTO retrain_logs (accuracy, record_count, training_source, timestamp) VALUES (?, ?, ?, ?)",
            (accuracy, record_count, source, current_time)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Log Error: {e}")

def get_analytics_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        def get_count(query):
            res = cursor.execute(query).fetchone()
            return res[0] if res else 0

        total = get_count("SELECT COUNT(*) FROM predictions")
        fake = get_count("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'")
        real = get_count("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'")
        flagged_count = get_count("SELECT COUNT(*) FROM predictions WHERE flagged='Yes'")

        # Flagged Posts List
        rows = cursor.execute("SELECT * FROM predictions WHERE flagged='Yes' ORDER BY id DESC LIMIT 5").fetchall()
        flagged_posts = [dict(row) for row in rows]

        # Daily Stats
        daily_data = cursor.execute("""
            SELECT DATE(timestamp) as day, 
                SUM(CASE WHEN prediction = 'Real Job' THEN 1 ELSE 0 END) as real_count,
                SUM(CASE WHEN prediction = 'Fake Job' THEN 1 ELSE 0 END) as fake_count
            FROM predictions 
            GROUP BY day 
            ORDER BY day DESC LIMIT 7
        """).fetchall()
        
        daily_data = daily_data[::-1]
        dates = [row['day'] for row in daily_data]
        real_counts = [row['real_count'] for row in daily_data]
        fake_counts = [row['fake_count'] for row in daily_data]

        conn.close()
        return total, fake, real, flagged_count, dates, real_counts, fake_counts, flagged_posts
    except Exception as e:
        print(f"Analytics Error: {e}")
        return 0, 0, 0, 0, [], [], [], []

def clean_text(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ==========================================
# PUBLIC ROUTES
# ==========================================

@app.route('/')
def home():
    recent_scans = []
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5").fetchall()
        conn.close()
        recent_scans = [dict(row) for row in rows]
    except:
        recent_scans = []
    return render_template('home.html', recent_scans=recent_scans)

@app.route('/scanner')
def scanner_page():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        raw_text = data.get('text', '')
        if not raw_text: return jsonify({'error': 'No text provided'}), 400

        cleaned_text = clean_text(raw_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        pred_val = model.predict(vectorized_text)[0]
        try:
            proba = model.predict_proba(vectorized_text)[0]
            confidence = round(max(proba) * 100, 2)
        except: confidence = 0.0

        result = "Fake Job" if pred_val == 1 else "Real Job"
        post_id = log_to_db(raw_text, result, confidence)

        return jsonify({'status': 'success', 'prediction': result, 'confidence': confidence, 'post_id': post_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/flag_post', methods=['POST'])
def flag_post():
    try:
        data = request.json
        post_id = data.get('post_id')
        conn = get_db_connection()
        conn.execute("UPDATE predictions SET flagged='Yes' WHERE id=?", (post_id,))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Post flagged for admin review.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# ADMIN ROUTES
# ==========================================

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.execute("SELECT * FROM admin WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['admin_logged_in'] = True
            return redirect('/admin_dashboard')
        else:
            return render_template('admin_login.html', error="Invalid Credentials")
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'): return redirect('/admin_login')
    
    # 1. Get Analytics
    total, fake, real, flagged_count, dates, real_counts, fake_counts, flagged_posts = get_analytics_db()
    
    # 2. Get Last Retrain Info (Task 2)
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retrain_logs'")
        if cursor.fetchone():
            last_retrain = conn.execute("SELECT * FROM retrain_logs ORDER BY id DESC LIMIT 1").fetchone()
        else:
            last_retrain = None
    except Exception:
        last_retrain = None
    conn.close()

    # Safety checks for NoneType
    if last_retrain:
        records_display = f"{last_retrain['record_count']:,}" if last_retrain['record_count'] is not None else "N/A"
        acc_display = last_retrain['accuracy'] if last_retrain['accuracy'] is not None else 0.0
        last_date_display = last_retrain['timestamp']
    else:
        records_display = "0"
        acc_display = 0.0
        last_date_display = "Never"

    model_info = {
        'last_date': last_date_display,
        'records': records_display,
        'accuracy': acc_display,
        'model_type': "Logistic Regression (Balanced)"
    }

    return render_template('admin_dashboard.html', 
                           fake=fake, real=real, total=total, flagged_count=flagged_count,
                           dates=dates, real_counts=real_counts, fake_counts=fake_counts,
                           flagged_posts=flagged_posts, 
                           model_info=model_info)

@app.route('/admin_history')
def admin_history():
    if not session.get('admin_logged_in'): return redirect('/admin_login')
    
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()

    history_data = [dict(row) for row in rows]
    return render_template('admin_history.html', history=history_data, page="history")

@app.route('/admin_action', methods=['POST'])
def admin_action():
    if not session.get('admin_logged_in'): return jsonify({'error': 'Unauthorized'}), 403
    try:
        data = request.json
        post_id = int(data.get('post_id'))
        action = data.get('action') 
        conn = get_db_connection()
        if action == 'delete':
            conn.execute("DELETE FROM predictions WHERE id=?", (post_id,))
            msg = "Post deleted successfully."
        elif action == 'dismiss':
            conn.execute("UPDATE predictions SET flagged='No' WHERE id=?", (post_id,))
            msg = "Flag removed."
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': msg})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/export_data')
def export_data():
    if not session.get('admin_logged_in'): return redirect('/admin_login')
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        output = io.StringIO()
        df.to_csv(output, index=False)
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-disposition": "attachment; filename=job_predictions_export.csv"})
    except Exception as e:
        return str(e)

@app.route('/training_logs')
def training_logs():
    if not session.get('admin_logged_in'): return redirect('/admin_login')
    
    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT * FROM retrain_logs ORDER BY id DESC").fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return render_template('training_logs.html', logs=rows)

# --- TASK 1: Compare Page Route (with Sidebar Support) ---
@app.route('/model_comparison')
def model_comparison():
    if not session.get('admin_logged_in'): return redirect('/admin_login')
    
    # 1. Try URL parameters (from immediate training)
    old = request.args.get('old')
    new = request.args.get('new')
    imp = request.args.get('imp')

    # 2. If Sidebar clicked (no params), fetch from DB
    if old is None or new is None:
        conn = get_db_connection()
        try:
            # Fetch last 2 logs
            rows = conn.execute("SELECT * FROM retrain_logs ORDER BY id DESC LIMIT 2").fetchall()
            
            if len(rows) >= 1:
                new_val = rows[0]['accuracy']
                # Normalize accuracy (0.xx vs xx.x)
                if new_val < 1: new_val *= 100
                
                if len(rows) >= 2:
                    old_val = rows[1]['accuracy']
                    if old_val < 1: old_val *= 100
                else:
                    old_val = 0.0

                imp_val = new_val - old_val
                
                new = round(new_val, 1)
                old = round(old_val, 1)
                imp = round(imp_val, 1)
            else:
                new, old, imp = 0, 0, 0
        except Exception as e:
            print(f"Error fetching comparison: {e}")
            new, old, imp = 0, 0, 0
        finally:
            conn.close()

    return render_template('model_comparison.html', old=old, new=new, imp=imp)

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    if not session.get('admin_logged_in'): return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        result = subprocess.run(["python", TRAIN_SCRIPT], capture_output=True, text=True)
        if result.returncode == 0:
            load_model_resources()
            
            output = result.stdout
            try:
                # Capture JSON output from train_model.py
                if "JSON_START" in output:
                    json_str = output.split("JSON_START")[1].split("JSON_END")[0].strip()
                    data = json.loads(json_str)
                    
                    # Store in DB with correct source name
                    log_retrain_event(data['new_accuracy'], data['record_count'], "fake_job_postings.csv")
                    
                    # Return data for Popup
                    return jsonify({
                        'status': 'success',
                        'redirect_url': url_for('model_comparison', 
                                                old=data['old_accuracy'], 
                                                new=data['new_accuracy'], 
                                                imp=data['improvement'])
                    })
                else:
                    return jsonify({'status': 'error', 'message': "Training finished but returned no JSON data."})
            except Exception as parse_err:
                print(f"Parse Error: {parse_err}")
                return jsonify({'status': 'success', 'message': 'Model Retrained (No Comparison Data Available)'})
        else:
            return jsonify({'status': 'error', 'message': f"Training Failed: {result.stderr}"})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    # Safety Check
    if not os.path.exists(DB_PATH):
        print(f"⚠️ Warning: Database not found at {DB_PATH}")
        print("   Please run 'create_db.py' first!")
    
    app.run(debug=True, port=5000)
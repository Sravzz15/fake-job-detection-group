import sqlite3
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "job_predictions.db")

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_description TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        flagged TEXT DEFAULT 'No'
    )
    """)
    
    # 2. Admin Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # 3. Retrain Logs Table (Updated for Task 1 & 2)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS retrain_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        accuracy REAL,
        record_count INTEGER,
        training_source TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Initial Data Insertion ---

    # Default Admin
    try:
        cursor.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ("admin", "admin123"))
        print("Default Admin created: admin / admin123")
    except sqlite3.IntegrityError:
        print("Admin already exists.")

    # Insert dummy log to prevent Dashboard Error
    cursor.execute("""
        INSERT INTO retrain_logs (accuracy, record_count, training_source) 
        VALUES (0.88, 10000, 'Initial Setup')
    """)
    print("Initial dummy training log added.")

    conn.commit()
    conn.close()
    print(f"Database created successfully at: {DB_PATH}")

if __name__ == "__main__":
    # Remove old DB if exists
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("Old database removed to update schema.")
        except Exception as e:
            print(f"Could not remove old DB: {e}")
            
    create_table()

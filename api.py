"""
Flask API for Mentor-Mentee Matching System
Simple REST API to serve predictions in production

Install: pip install flask flask-cors
Run: python api.py
Test: curl http://localhost:5000/api/health
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration


@app.before_request
def _log_incoming_request():
    try:
        print(f"Incoming request -> {request.method} {request.path}")
    except Exception:
        pass

# Configuration
DB_URL = os.getenv('DATABASE_URL', "postgresql://aihak:your_password_here@localhost:5432/mentor_ai")
MODEL_PATH = 'mentor_matching_model.pkl'
FEATURE_COLS_PATH = 'feature_columns.pkl'

# Global variables for model and data (loaded once at startup)
model = None
feature_cols = None
mentors_df = None
mentees_df = None

# Feature engineering functions
def calculate_domain_overlap(mentor_domains, mentee_domains):
    if not mentor_domains or not mentee_domains:
        return 0
    overlap = len(set(mentor_domains) & set(mentee_domains))
    return overlap / len(mentee_domains) if mentee_domains else 0

def calculate_skill_overlap(mentor_skills, mentee_skills):
    if not mentor_skills or not mentee_skills:
        return 0
    overlap = len(set(mentor_skills) & set(mentee_skills))
    total = len(set(mentor_skills) | set(mentee_skills))
    return overlap / total if total > 0 else 0

def calculate_availability_compatibility(mentor_hours, mentee_hours):
    diff = abs(mentor_hours - mentee_hours)
    return max(0, 1 - (diff / 30))

def style_match(mentor_style, mentee_style):
    return 1 if mentor_style == mentee_style else 0



def industry_match(mentor_industry, mentee_industry):
    return 1 if mentor_industry == mentee_industry else 0

def experience_level_compatibility(experience_years, current_level):
    level_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    mentee_level_numeric = level_map.get(current_level, 2)
    
    if mentee_level_numeric == 1 and experience_years >= 10:
        return 1.0
    elif mentee_level_numeric == 2 and experience_years >= 7:
        return 0.8
    elif mentee_level_numeric == 3 and experience_years >= 5:
        return 0.6
    return 0.3

def load_resources():
    """Load model and data at startup"""
    global model, feature_cols, mentors_df, mentees_df
    
    print("Loading resources...")
    
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        print(f"✓ Model loaded: {type(model).__name__}")
    except FileNotFoundError:
        print(" Model files not found. Train model first: python ml_pipeline.py")
        return False
    
    # Load data
    try:
        engine = create_engine(DB_URL)
        
        with engine.connect() as conn:
            mentors_df = pd.read_sql_query(text("SELECT * FROM mentors WHERE active = TRUE"), conn)
            mentees_df = pd.read_sql_query(text("SELECT * FROM mentees "), conn)
        
        print(f"✓ Loaded {len(mentors_df)} mentors, {len(mentees_df)} mentees")
        # Debug: print mentee id column info to help diagnose 404 on /api/recommend
        try:
            print("Mentees dataframe columns:", list(mentees_df.columns))
            if 'mentee_id' in mentees_df.columns:
                print("Sample mentee_id values:", mentees_df['mentee_id'].head(30).tolist())
                print("mentee_id dtype:", mentees_df['mentee_id'].dtype)
            else:
                print("Warning: 'mentee_id' column not found in mentees table")
        except Exception as _:
            pass
    except Exception as e:
        print(f" Database connection failed: {e}")
        return False
    
    return True

def create_features(mentor, mentee):
    """Create feature vector for a mentor-mentee pair"""
    features = {
        'domain_overlap': calculate_domain_overlap(mentor['domains'], mentee['desired_domains']),
        'skill_overlap': calculate_skill_overlap(mentor['skills'], mentee.get('current_skills', [])),
        'availability_compatibility': calculate_availability_compatibility(
            mentor['availability_hours'], mentee['availability_hours']
        ),
        'style_match': style_match(mentor['mentorship_style'], mentee['preferred_style']),
        'industry_match': industry_match(mentor['industry'], mentee['industry']),
        'mentor_experience_years': mentor['experience_years'],
        'mentor_rating': mentor['rating'],
        'mentor_acceptance_rate': mentor['acceptance_rate'],
        'mentor_total_mentees': mentor['total_mentees'],
        'mentee_level_numeric': {'beginner': 1, 'intermediate': 2, 'advanced': 3}.get(
            mentee['current_level'], 2
        ),
        'experience_compatibility': experience_level_compatibility(
            mentor['experience_years'], mentee['current_level']
        ),
        'mentor_domain_count': len(mentor['domains']) if mentor['domains'] else 0,
        'mentee_domain_count': len(mentee['desired_domains']) if mentee['desired_domains'] else 0,
        'mentor_skill_count': len(mentor['skills']) if mentor['skills'] else 0,
    }
    return features

# API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mentors_count': len(mentors_df) if mentors_df is not None else 0,
        'mentees_count': len(mentees_df) if mentees_df is not None else 0
    })

@app.route('/api/mentors', methods=['GET'])
def get_mentors():
    """Get all active mentors"""
    if mentors_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Convert to dict and handle arrays
    mentors_list = mentors_df.to_dict('records')
    for mentor in mentors_list:
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(mentor.get('skills'), np.ndarray):
            mentor['skills'] = mentor['skills'].tolist()
        if isinstance(mentor.get('domains'), np.ndarray):
            mentor['domains'] = mentor['domains'].tolist()
    
    return jsonify({
        'count': len(mentors_list),
        'mentors': mentors_list
    })

@app.route('/api/mentees', methods=['GET'])
def get_mentees():
    """Get all active mentees"""
    if mentees_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    mentees_list = mentees_df.to_dict('records')
    for mentee in mentees_list:
        if isinstance(mentee.get('current_skills'), np.ndarray):
            mentee['current_skills'] = mentee['current_skills'].tolist()
        if isinstance(mentee.get('desired_domains'), np.ndarray):
            mentee['desired_domains'] = mentee['desired_domains'].tolist()
    
    return jsonify({
        'count': len(mentees_list),
        'mentees': mentees_list
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_mentors():
    """
    Recommend mentors for a mentee
    
    POST /api/recommend
    Body: {
        "mentee_id": 123,
        "top_k": 5  (optional, default 5)
    }
    """
    if model is None or mentors_df is None or mentees_df is None:
        return jsonify({'error': 'Resources not loaded'}), 500
    
    data = request.get_json()
    
    if not data or 'mentee_id' not in data:
        return jsonify({'error': 'mentee_id is required'}), 400
    
    mentee_id = data['mentee_id']
    top_k = data.get('top_k', 5)
    
    # Get mentee
    mentee = mentees_df[mentees_df['mentee_id'] == mentee_id]
    if len(mentee) == 0:
        return jsonify({'error': f'Mentee {mentee_id} not found'}), 404
    
    mentee = mentee.iloc[0]
    
    # Generate recommendations
    recommendations = []
    
    for _, mentor in mentors_df.iterrows():
        features = create_features(mentor, mentee)
        feature_vector = pd.DataFrame([features])[feature_cols]
        success_prob = model.predict_proba(feature_vector)[0][1]
        
        recommendations.append({
            'mentor_id': int(mentor['mentor_id']),
            'mentor_name': mentor['name'],
            'match_score': float(success_prob),
            'domains': mentor['domains'].tolist() if isinstance(mentor['domains'], np.ndarray) else mentor['domains'],
            'experience_years': int(mentor['experience_years']),
            'rating': float(mentor['rating']),
            'mentorship_style': mentor['mentorship_style'],
            'availability_hours': int(mentor['availability_hours']),
            'bio': mentor['bio'],
            'email': mentor['email'],
            'match_details': {
                'domain_overlap': float(features['domain_overlap']),
                'style_match': bool(features['style_match']),
                'availability_compatibility': float(features['availability_compatibility'])
            }
        })
    
    # Sort by match score and return top K
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    top_recommendations = recommendations[:top_k]
    
    return jsonify({
        'mentee_id': int(mentee_id),
        'mentee_name': mentee['name'],
        'recommendations_count': len(top_recommendations),
        'recommendations': top_recommendations
    })

@app.route('/api/match-score', methods=['POST'])
def get_match_score():
    """
    Get match score for a specific mentor-mentee pair
    
    POST /api/match-score
    Body: {
        "mentor_id": 123,
        "mentee_id": 456
    }
    """
    if model is None or mentors_df is None or mentees_df is None:
        return jsonify({'error': 'Resources not loaded'}), 500
    
    data = request.get_json()
    
    if not data or 'mentor_id' not in data or 'mentee_id' not in data:
        return jsonify({'error': 'Both mentor_id and mentee_id are required'}), 400
    
    mentor_id = data['mentor_id']
    mentee_id = data['mentee_id']
    
    # Get mentor and mentee
    mentor = mentors_df[mentors_df['mentor_id'] == mentor_id]
    mentee = mentees_df[mentees_df['mentee_id'] == mentee_id]
    
    if len(mentor) == 0:
        return jsonify({'error': f'Mentor {mentor_id} not found'}), 404
    if len(mentee) == 0:
        return jsonify({'error': f'Mentee {mentee_id} not found'}), 404
    
    mentor = mentor.iloc[0]
    mentee = mentee.iloc[0]
    
    # Calculate match score
    features = create_features(mentor, mentee)
    feature_vector = pd.DataFrame([features])[feature_cols]
    success_prob = model.predict_proba(feature_vector)[0][1]
    
    return jsonify({
        'mentor_id': int(mentor_id),
        'mentor_name': mentor['name'],
        'mentee_id': int(mentee_id),
        'mentee_name': mentee['name'],
        'match_score': float(success_prob),
        'match_details': {
            'domain_overlap': float(features['domain_overlap']),
            'skill_overlap': float(features['skill_overlap']),
            'style_match': bool(features['style_match']),
            'industry_match': bool(features['industry_match']),
            'availability_compatibility': float(features['availability_compatibility']),
            'experience_compatibility': float(features['experience_compatibility'])
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*60)
    print("MENTOR-MENTEE MATCHING API")
    print("="*60)
    
    # Load resources
    if not load_resources():
        print("\n Failed to load resources. Exiting.")
        exit(1)
    
    print("\n✓ API ready!")
    print("\nEndpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/mentors         - List all mentors")
    print("  GET  /api/mentees         - List all mentees")
    print("  POST /api/recommend       - Get mentor recommendations")
    print("  POST /api/match-score     - Get match score for a pair")
    
    print("\n" + "="*60)
    print("Starting server on http://localhost:5000")
    print("="*60)
    # Print Flask URL map to show registered routes
    try:
        print("\nRegistered URL routes:")
        print(app.url_map)
    except Exception:
        pass
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)

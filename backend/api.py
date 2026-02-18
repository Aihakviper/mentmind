

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import joblib
import os
from functools import lru_cache
import time

app = Flask(__name__)
CORS(app)


cache_config = {
    'CACHE_TYPE': 'SimpleCache',  
    'CACHE_DEFAULT_TIMEOUT': 300  
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Configuration
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv('DATABASE_URL')
if not DB_URL:
    raise RuntimeError("DATABASE_URL not set. Add it to .env or environment variables before starting the API.")
MODEL_PATH = 'mentor_matching_model.pkl'
FEATURE_COLS_PATH = 'feature_columns.pkl'

# Global variables for model and data 
model = None
feature_cols = None
mentors_df = None
mentees_df = None


engine = None

# Feature engineering functions 
def calculate_domain_overlap(mentor_domains, mentee_domains):
    if not mentor_domains or not mentee_domains:
        return 0.0
    overlap = len(set(mentor_domains) & set(mentee_domains))
    return float(overlap) / len(mentee_domains)

def calculate_skill_overlap(mentor_skills, mentee_skills):
    if not mentor_skills or not mentee_skills:
        return 0.0
    overlap = len(set(mentor_skills) & set(mentee_skills))
    total = len(set(mentor_skills) | set(mentee_skills))
    return float(overlap) / total if total > 0 else 0.0

def calculate_availability_compatibility(mentor_hours, mentee_hours):
    diff = abs(mentor_hours - mentee_hours)
    return max(0.0, 1.0 - (diff / 30.0))

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

# Optimized feature creation using vectorization where possible
def create_features_batch(mentee, mentors_subset):
    """Create features for multiple mentors at once (faster)"""
    features_list = []
    
    for _, mentor in mentors_subset.iterrows():
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
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def load_resources():
    """Load model and data at startup with connection pooling"""
    global model, feature_cols, mentors_df, mentees_df, engine
    
    print("Loading resources...")
    start_time = time.time()
    
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        print(f" Model loaded: {type(model).__name__}")
    except FileNotFoundError:
        print(" Model files not found. Train model first: python ml_pipeline.py")
        return False
    
    # Create database engine with connection pooling
    try:
        engine = create_engine(
            DB_URL,
            poolclass=QueuePool,
            pool_size=10,        
            max_overflow=20,    
            pool_pre_ping=True,  
            pool_recycle=3600   
        )
        
        # Load data into memory for faster access
        with engine.connect() as conn:
            mentors_df = pd.read_sql_query(
                text("SELECT * FROM mentors WHERE active = TRUE"), 
                conn
            )
            mentees_df = pd.read_sql_query(
                text("SELECT * FROM mentees WHERE active = TRUE"), 
                conn
            )
        
        # Convert to optimal data types
        mentors_df['mentor_id'] = mentors_df['mentor_id'].astype('int32')
        mentees_df['mentee_id'] = mentees_df['mentee_id'].astype('int32')
        
        elapsed = time.time() - start_time
        print(f" Loaded {len(mentors_df)} mentors, {len(mentees_df)} mentees in {elapsed:.2f}s")
        
    except Exception as e:
        print(f" Database connection failed: {e}")
        return False
    
    return True

# API Endpoints 

@app.route('/api/health', methods=['GET'])
@cache.cached(timeout=60)  
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mentors_count': len(mentors_df) if mentors_df is not None else 0,
        'mentees_count': len(mentees_df) if mentees_df is not None else 0
    })

@app.route('/api/mentors', methods=['GET'])
@cache.cached(timeout=300)  
def get_mentors():
    """Get all active mentors (cached)"""
    if mentors_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Convert to dict 
    mentors_list = mentors_df.to_dict('records')
    
    # Handle numpy arrays
    for mentor in mentors_list:
        if isinstance(mentor.get('skills'), np.ndarray):
            mentor['skills'] = mentor['skills'].tolist()
        if isinstance(mentor.get('domains'), np.ndarray):
            mentor['domains'] = mentor['domains'].tolist()
    
    return jsonify({
        'count': len(mentors_list),
        'mentors': mentors_list
    })

@app.route('/api/mentees', methods=['GET'])
@cache.cached(timeout=300)
def get_mentees():
    """Get all active mentees (cached)"""
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
    Get mentor recommendations (optimized with batch processing)
    Cache based on mentee_id and top_k
    """
    if model is None or mentors_df is None or mentees_df is None:
        return jsonify({'error': 'Resources not loaded'}), 500
    
    start_time = time.time()
    
    data = request.get_json()
    
    if not data or 'mentee_id' not in data:
        return jsonify({'error': 'mentee_id is required'}), 400
    
    mentee_id = data['mentee_id']
    top_k = data.get('top_k', 5)
    
    # Cache key based on request
    cache_key = f"recommend_{mentee_id}_{top_k}"
    
    # Check cache first
    cached_result = cache.get(cache_key)
    if cached_result:
        cached_result['cached'] = True
        cached_result['response_time'] = f"{(time.time() - start_time) * 1000:.2f}ms"
        return jsonify(cached_result)
    
    # Get mentee
    mentee = mentees_df[mentees_df['mentee_id'] == mentee_id]
    if len(mentee) == 0:
        return jsonify({'error': f'Mentee {mentee_id} not found'}), 404
    
    mentee = mentee.iloc[0]
    
    # Batch process all mentors at once 
    features_df = create_features_batch(mentee, mentors_df)
    feature_matrix = features_df[feature_cols].values
    
    # Single batch prediction (faster than individual predictions)
    success_probs = model.predict_proba(feature_matrix)[:, 1]
    
    # Get top K indices
    top_indices = np.argpartition(success_probs, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(success_probs[top_indices])[::-1]]
    
    # Build recommendations
    recommendations = []
    for idx in top_indices:
        mentor = mentors_df.iloc[idx]
        features = features_df.iloc[idx]
        
        recommendations.append({
            'mentor_id': int(mentor['mentor_id']),
            'mentor_name': mentor['name'],
            'match_score': float(success_probs[idx]),
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
    
    result = {
        'mentee_id': int(mentee_id),
        'mentee_name': mentee['name'],
        'recommendations_count': len(recommendations),
        'recommendations': recommendations,
        'cached': False,
        'response_time': f"{(time.time() - start_time) * 1000:.2f}ms"
    }
    
    # Cache the result
    cache.set(cache_key, result, timeout=300)  
    
    return jsonify(result)

@app.route('/api/match-score', methods=['POST'])
@cache.memoize(timeout=300) 
def get_match_score():
    """Get match score for a specific mentor-mentee pair (cached)"""
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
    
    # Calculate features
    features_df = create_features_batch(mentee, mentors_df[mentors_df['mentor_id'] == mentor_id])
    features = features_df.iloc[0]
    
    # Predict
    feature_vector = features_df[feature_cols]
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

# Clear cache endpoint 
@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all cached data"""
    cache.clear()
    return jsonify({'message': 'Cache cleared successfully'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*60)
    print("OPTIMIZED MENTOR-MENTEE MATCHING API")
    print("="*60)
    
    # Load resources
    if not load_resources():
        print("\n Failed to load resources. Exiting.")
        exit(1)
    
   
    
    # Run server with optimized settings
    app.run(
        debug=False,         
        host='0.0.0.0',
        port=5000,
        threaded=True        

    )

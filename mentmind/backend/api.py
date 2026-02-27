"""
Flask API for Mentor-Mentee Matching System
Supports recommending mentors using input mentee data (no DB required for /api/recommend)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
import numpy as np
import joblib
import os
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address



app = Flask(__name__)
CORS(app)

cache_config = {
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
}
app.config.from_mapping(cache_config)
cache = Cache(app)
limiter = Limiter(get_remote_address, app=app)

# Configuration

from dotenv import load_dotenv
load_dotenv()


DB_URL = os.getenv('DATABASE_URL')          
MODEL_PATH        = os.getenv('MODEL_PATH',        'C:\\Users\\Dell\\Documents\\recommendation system\\model\\mentor_matching_modelv2.pkl')
FEATURE_COLS_PATH = os.getenv('FEATURE_COLS_PATH', 'C:\\Users\\Dell\\Documents\\recommendation system\\model\\feature_columnsv2.pkl')
TFIDF_PATH        = os.getenv('TFIDF_PATH',        'C:\\Users\\Dell\\Documents\\recommendation system\\model\\tfidf_vectorizerv2.pkl')
SBERT_VECS_PATH   = os.getenv('SBERT_VECS_PATH',   'C:\\Users\\Dell\\Documents\\recommendation system\\model\\mentor_sbert_vecsv2.pkl') 

# Global state
model = None
feature_cols = None
tfidf_vectorizer   = None
mentor_sbert_vecs  = None
sbert_model = None
mentors_df = None   # Loaded from DB at startup (if DB available)
mentees_df = None   # Loaded from DB at startup (if DB available)
engine = None

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

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
    try:
        diff = abs(float(mentor_hours) - float(mentee_hours))
        return max(0.0, 1.0 - (diff / 30.0))
    except (TypeError, ValueError):
        return 0.5  # neutral fallback

def style_match(mentor_style, mentee_style):
    return 1 if mentor_style == mentee_style else 0

def industry_match(mentor_industry, mentee_industry):
    return 1 if mentor_industry == mentee_industry else 0

def experience_level_compatibility(experience_years, current_level):
    level_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    mentee_level_numeric = level_map.get(current_level, 2)
    try:
        exp = float(experience_years)
    except (TypeError, ValueError):
        return 0.3
    if mentee_level_numeric == 1 and exp >= 10:
        return 1.0
    elif mentee_level_numeric == 2 and exp >= 7:
        return 0.8
    elif mentee_level_numeric == 3 and exp >= 5:
        return 0.6
    return 0.3

from sklearn.metrics.pairwise import cosine_similarity

def bio_goal_tfidf_sim(mentor_bio: str, mentee_goals: str) -> float:
    if tfidf_vectorizer is None or not mentor_bio or not mentee_goals:
        return 0.0
    m_vec = tfidf_vectorizer.transform([mentor_bio])
    e_vec = tfidf_vectorizer.transform([mentee_goals])
    return float(cosine_similarity(m_vec, e_vec)[0, 0])

def bio_goal_sbert_sim(mentor_idx: int, mentee_goals: str) -> float:
    """Uses pre-computed mentor SBERT vecs; encodes mentee goals on the fly."""
    if mentor_sbert_vecs is None or not mentee_goals:
        return bio_goal_tfidf_sim(
            mentors_df.iloc[mentor_idx].get('bio', ''), mentee_goals
        )
    try:
        from sentence_transformers import SentenceTransformer
        _sbert =  sbert_model if sbert_model else SentenceTransformer('all-MiniLM-L6-v2')
        e_emb  = _sbert.encode([mentee_goals], convert_to_numpy=True)
        return float(cosine_similarity(
            mentor_sbert_vecs[mentor_idx].reshape(1, -1), e_emb)[0, 0])
    except Exception:
        return bio_goal_tfidf_sim(
            mentors_df.iloc[mentor_idx].get('bio', ''), mentee_goals
        )

def bio_domain_coverage(mentor_bio: str, mentee_domains: list) -> float:
    if not mentor_bio or not mentee_domains:
        return 0.0
    bio_lower = mentor_bio.lower()
    return sum(1 for d in mentee_domains if d.lower() in bio_lower) / len(mentee_domains)

def bio_skill_coverage(mentor_bio: str, mentee_skills: list) -> float:
    if not mentor_bio or not mentee_skills:
        return 0.0
    bio_lower = mentor_bio.lower()
    return sum(1 for s in mentee_skills if s.lower() in bio_lower) / len(mentee_skills)

def experience_gap_score(experience_years: float, current_level: str) -> float:
    ideal = {'beginner': (8, 15), 'intermediate': (5, 12), 'advanced': (3, 8)}
    lo, hi = ideal.get(current_level, (5, 12))
    if lo <= experience_years <= hi:  return 1.0
    elif experience_years < lo:       return max(0.0, experience_years / lo)
    else:                             return max(0.4, 1.0 - (experience_years - hi) / 20)

def create_features_batch(mentee, mentors_subset):
    """Build feature DataFrame for every mentor in mentors_subset vs. one mentee."""
    features_list = []
    level_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    mentee_level_numeric = level_map.get(mentee.get('current_level'), 2)
    mentee_goals = str(mentee.get('goals', ''))
    

       

    for pos, mentor in mentors_subset.iterrows():
        mentor_bio    = str(mentor.get('bio', ''))
        mentor_domains = mentor.get('domains') or []
        mentor_skills  = mentor.get('skills')  or []
        if isinstance(mentor_domains, np.ndarray):
            mentor_domains = mentor_domains.tolist()
        if isinstance(mentor_skills, np.ndarray):
            mentor_skills = mentor_skills.tolist()

        features = {
            'domain_overlap': calculate_domain_overlap(
                mentor_domains, mentee.get('desired_domains', [])
            ),
            'skill_overlap': calculate_skill_overlap(
                mentor_skills, mentee.get('current_skills', [])
            ),
            'availability_compat': calculate_availability_compatibility(
                mentor.get('availability_hours', 10),
                mentee.get('availability_hours', 5)
            ),
            'style_match': style_match(
                mentor.get('mentorship_style'), mentee.get('preferred_style')
            ),
            'industry_match': industry_match(
                mentor.get('industry'), mentee.get('industry')
            ),
            'mentor_experience_years': mentor.get('experience_years', 0),
            'experience_gap_score':  experience_gap_score(
                                         float(mentor.get('experience_years', 0)),
                                         mentee.get('current_level', 'intermediate')),
            'mentee_level_numeric':    mentee_level_numeric,
            'mentor_rating':           mentor.get('rating', 3.0),
            'mentor_acceptance_rate':  mentor.get('acceptance_rate', 0.5),
            'mentor_total_mentees':    mentor.get('total_mentees', 0),
            'mentor_domain_count': len(mentor_domains),
            'mentee_domain_count': len(mentee.get('desired_domains') or []),
            'mentor_skill_count':  len(mentor_skills),
            'mentee_skill_count':    len(mentee.get('current_skills') or []),
            
            'bio_goal_tfidf_sim':    bio_goal_tfidf_sim(mentor_bio, mentee_goals),
            'bio_goal_sbert_sim':    bio_goal_sbert_sim(pos, mentee_goals),
            'bio_domain_coverage':   bio_domain_coverage(mentor_bio,
                                         mentee.get('desired_domains') or []),
            'bio_skill_coverage':    bio_skill_coverage(mentor_bio,
                                         mentee.get('current_skills') or []),
        }
        features_list.append(features)

    return pd.DataFrame(features_list)


# Startup: load model (required) + DB data (optional)


def load_resources():
    global model, feature_cols, tfidf_vectorizer, mentor_sbert_vecs, sbert_model, mentors_df, mentees_df, engine

    print("Loading resources...")

    # ---- Model (required) ----
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        print(f"   Model loaded: {type(model).__name__}")
    except FileNotFoundError as e:
        print(f"   Model files not found: {e}")
        print("    Train your model first, then update MODEL_PATH in .env or the script.")
        return False
        
    try:
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        print(f"   TF-IDF vectorizer loaded  (vocab: {len(tfidf_vectorizer.vocabulary_)})")
    except FileNotFoundError:
        print("   tfidf_vectorizerv2.pkl not found — NLP features will be 0")

    try:
        mentor_sbert_vecs = joblib.load(SBERT_VECS_PATH)
        print(f"   SBERT vectors loaded  shape={mentor_sbert_vecs.shape}")
    except FileNotFoundError:
        print("   mentor_sbert_vecs.pkl not found — will fall back to TF-IDF")
        
    if mentor_sbert_vecs is not None:
        try:
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("   SBERT model loaded")
        except ImportError:
            print("   sentence-transformers not installed — SBERT disabled")
    # ---- Database (optional) ----
    if not DB_URL:
        print("    DATABASE_URL not set — DB-dependent endpoints will be unavailable.")
        print("     /api/recommend with mentee_data will still work.")
        return True   

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import QueuePool

        engine = create_engine(
            DB_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        with engine.connect() as conn:
            mentors_df  = pd.read_sql_query(text("SELECT * FROM mentors  WHERE active = TRUE"), conn)
            mentees_df  = pd.read_sql_query(text("SELECT * FROM mentees  WHERE active = TRUE"), conn)

        mentors_df['mentor_id'] = mentors_df['mentor_id'].astype('int32')
        mentees_df['mentee_id'] = mentees_df['mentee_id'].astype('int32')
        print(f"   Loaded {len(mentors_df)} mentors, {len(mentees_df)} mentees from DB")
    except Exception as e:
        print(f"    Database connection failed: {e}")
        print("     /api/recommend with mentee_data will still work if you pass mentors in the request.")

    return True


# Endpoints


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'db_connected': mentors_df is not None,
        'mentors_count': len(mentors_df) if mentors_df is not None else 0,
        'mentees_count': len(mentees_df) if mentees_df is not None else 0,
    })


@app.route('/api/mentors', methods=['GET'])
@cache.cached(timeout=300)
def get_mentors():
    if mentors_df is None:
        return jsonify({'error': 'Mentor data not loaded (DB unavailable)'}), 503

    mentors_list = mentors_df.to_dict('records')
    for m in mentors_list:
        if isinstance(m.get('skills'),  np.ndarray): m['skills']  = m['skills'].tolist()
        if isinstance(m.get('domains'), np.ndarray): m['domains'] = m['domains'].tolist()

    return jsonify({'count': len(mentors_list), 'mentors': mentors_list})


@app.route('/api/mentees', methods=['GET'])
@cache.cached(timeout=300)
def get_mentees():
    if mentees_df is None:
        return jsonify({'error': 'Mentee data not loaded (DB unavailable)'}), 503

    mentees_list = mentees_df.to_dict('records')
    for m in mentees_list:
        if isinstance(m.get('current_skills'),  np.ndarray): m['current_skills']  = m['current_skills'].tolist()
        if isinstance(m.get('desired_domains'), np.ndarray): m['desired_domains'] = m['desired_domains'].tolist()

    return jsonify({'count': len(mentees_list), 'mentees': mentees_list})


@app.route('/api/recommend', methods=['POST'])
@limiter.limit("10 per minute")
def recommend_mentors():
    """
    Recommend mentors for a mentee.

    Two modes:
      A) mentee_data  — pass mentee profile directly (no DB needed).
                        Optionally pass a 'mentors' list to use custom mentors
                        instead of the DB-loaded ones.
      B) mentee_id    — look up an existing mentee from the DB.

    Request body examples:

      Mode A (from form / registration):
        {
          "mentee_data": {
            "name": "Amina",
            "desired_domains": ["AI", "Product Management"],
            "current_skills": ["Python", "Excel"],
            "current_level": "intermediate",
            "industry": "Tech",
            "preferred_style": "structured",
            "availability_hours": 8
          },
          "top_k": 5
        }

      Mode B (existing mentee):
        { "mentee_id": 42, "top_k": 5 }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400

    start_time = time.time()

   
    # 1. Resolve mentee

    if 'mentee_data' in data:
        mentee = data['mentee_data']
        mentee_name = mentee.get('name', 'New User')

    elif 'mentee_id' in data:
        if mentees_df is None:
            return jsonify({'error': 'DB not available; cannot look up mentee_id. Use mentee_data instead.'}), 503
        mentee_row = mentees_df[mentees_df['mentee_id'] == data['mentee_id']]
        if len(mentee_row) == 0:
            return jsonify({'error': f"Mentee {data['mentee_id']} not found"}), 404
        mentee = mentee_row.iloc[0].to_dict()
        mentee_name = mentee.get('name', str(data['mentee_id']))

    else:
        return jsonify({'error': 'Provide either "mentee_data" (dict) or "mentee_id" (int)'}), 400

    
    # 2. Resolve mentor pool

    if 'mentors' in data:
        
        pool = pd.DataFrame(data['mentors'])
    elif mentors_df is not None:
        pool = mentors_df
    else:
        return jsonify({
            'error': 'No mentor data available. '
                     'Either connect the DB, or pass a "mentors" list in the request body.'
        }), 503

    if pool.empty:
        return jsonify({'error': 'Mentor pool is empty'}), 400

  
    # 3. Build features & predict
    
    features_df = create_features_batch(mentee, pool)

    # Validate feature columns
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        return jsonify({'error': f'Missing feature columns: {missing}'}), 500

    feature_matrix = features_df[feature_cols].values
    success_probs  = model.predict_proba(feature_matrix)[:, 1]

    top_k       = int(data.get('top_k', 5))
    top_indices = np.argsort(success_probs)[::-1][:top_k]

    # 4. Build response
   
    recommendations = []
    for idx in top_indices:
        mentor = pool.iloc[idx]
        domains = mentor.get('domains') or []
        if isinstance(domains, np.ndarray):
            domains = domains.tolist()

        recommendations.append({
            'mentor_id':          int(mentor['mentor_id']) if 'mentor_id' in mentor else idx,
            'mentor_name':        str(mentor.get('name', 'Unknown')),
            'match_score':        float(success_probs[idx]),
            'industry':           str(mentor.get('industry', '')),
            'domains':            domains,
            'experience_years':   int(mentor.get('experience_years', 0)),
            'rating':             float(mentor.get('rating', 0.0)),
            'availability_hours': int(mentor.get('availability_hours', 0)),
            'mentorship_style':   str(mentor.get('mentorship_style', '')),
            'bio':                str(mentor.get('bio', '')),
            'email':              str(mentor.get('email', '')),
            'match_details': {
                'domain_overlap':       float(features_df.iloc[idx]['domain_overlap']),
                'skill_overlap':        float(features_df.iloc[idx]['skill_overlap']),
                'style_match':          bool(features_df.iloc[idx]['style_match']),
                'industry_match':       bool(features_df.iloc[idx]['industry_match']),
                'availability_compat':  float(features_df.iloc[idx]['availability_compat']),
                'experience_gap_score': float(features_df.iloc[idx]['experience_gap_score']),
                'bio_tfidf_sim':        float(features_df.iloc[idx]['bio_goal_tfidf_sim']),
                'bio_sbert_sim':        float(features_df.iloc[idx]['bio_goal_sbert_sim']),
                'bio_domain_coverage':  float(features_df.iloc[idx]['bio_domain_coverage']),
                'bio_skill_coverage':   float(features_df.iloc[idx]['bio_skill_coverage']),
            }
        })

    return jsonify({
        'mentee_name':           mentee_name,
        'recommendations_count': len(recommendations),
        'recommendations':       recommendations,
        'response_time':         f"{(time.time() - start_time) * 1000:.2f}ms"
    })


@app.route('/api/match-score', methods=['POST'])
def get_match_score():
    """Get match score for a specific mentor-mentee pair (both must be in DB)."""
    if model is None or mentors_df is None or mentees_df is None:
        return jsonify({'error': 'Resources not loaded (model or DB unavailable)'}), 503

    data = request.get_json(silent=True)
    if not data or 'mentor_id' not in data or 'mentee_id' not in data:
        return jsonify({'error': 'Both mentor_id and mentee_id are required'}), 400

    mentor_row = mentors_df[mentors_df['mentor_id'] == data['mentor_id']]
    mentee_row = mentees_df[mentees_df['mentee_id'] == data['mentee_id']]

    if mentor_row.empty:
        return jsonify({'error': f"Mentor {data['mentor_id']} not found"}), 404
    if mentee_row.empty:
        return jsonify({'error': f"Mentee {data['mentee_id']} not found"}), 404

    mentor = mentor_row.iloc[0]
    mentee = mentee_row.iloc[0].to_dict()

    features_df    = create_features_batch(mentee, mentor_row)
    feature_vector = features_df[feature_cols]
    success_prob   = model.predict_proba(feature_vector)[0][1]
    features       = features_df.iloc[0]

    return jsonify({
        'mentor_id':   int(data['mentor_id']),
        'mentor_name': str(mentor.get('name', '')),
        'mentee_id':   int(data['mentee_id']),
        'mentee_name': str(mentee.get('name', '')),
        'match_score': float(success_prob),
        'match_details': {
            'domain_overlap':             float(features['domain_overlap']),
            'skill_overlap':              float(features['skill_overlap']),
            'style_match':                bool(features['style_match']),
            'industry_match':             bool(features['industry_match']),
            'availability_compatibility': float(features['availability_compatibility']),
            'experience_compatibility':   float(features['experience_compatibility']),
        }
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    cache.clear()
    return jsonify({'message': 'Cache cleared'})


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
 
    print("MENTOR-MENTEE MATCHING API")
 

    if not load_resources():
        print("\n Failed to load model. Exiting.")
        exit(1)



    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

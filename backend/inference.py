"""
Inference Script: Make predictions with trained model
Use this to recommend mentors for mentees
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib
import sys

# Configuration
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL not set. Add it to .env or environment variables before running inference.")

# Feature engineering functions (same as training)
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

def timezone_match(mentor_tz, mentee_tz):
    return 1 if mentor_tz == mentee_tz else 0

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

def load_model_and_data():
    """Load trained model and database data"""
    print("Loading model and data...")
    
    # Load model
    try:
        model = joblib.load('mentor_matching_model.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        print("✓ Model loaded")
    except FileNotFoundError:
        print("❌ Model files not found. Train the model first:")
        print("   python ml_pipeline.py")
        sys.exit(1)
    
    # Load data
    engine = create_engine(DB_URL)
    
    with engine.connect() as conn:
        mentors_df = pd.read_sql_query(text("SELECT * FROM mentors WHERE active = TRUE"), conn)
        mentees_df = pd.read_sql_query(text("SELECT * FROM mentees WHERE active = TRUE"), conn)
    
    print(f"✓ Loaded {len(mentors_df)} mentors, {len(mentees_df)} mentees")
    
    return model, feature_cols, mentors_df, mentees_df

def recommend_mentors(mentee_id, model, feature_cols, mentors_df, mentees_df, top_k=5):
    """
    Recommend top K mentors for a given mentee
    
    Args:
        mentee_id: ID of the mentee
        model: Trained model
        feature_cols: List of feature column names
        mentors_df: DataFrame of mentors
        mentees_df: DataFrame of mentees
        top_k: Number of recommendations
    
    Returns:
        DataFrame with recommendations
    """
    
    # Get mentee
    mentee = mentees_df[mentees_df['mentee_id'] == mentee_id]
    if len(mentee) == 0:
        print(f"❌ Mentee ID {mentee_id} not found")
        return None
    
    mentee = mentee.iloc[0]
    
    print(f"\n{'='*80}")
    print(f"FINDING MENTORS FOR: {mentee['name']}")
    print(f"{'='*80}")
    print(f"Goals: {mentee['goals']}")
    print(f"Desired Domains: {mentee['desired_domains']}")
    print(f"Current Level: {mentee['current_level']}")
    print(f"Availability: {mentee['availability_hours']} hours/month")
    
    # Generate features for all mentors
    recommendations = []
    
    for _, mentor in mentors_df.iterrows():
        features = {
            'domain_overlap': calculate_domain_overlap(mentor['domains'], mentee['desired_domains']),
            'skill_overlap': calculate_skill_overlap(mentor['skills'], mentee.get('current_skills', [])),
            'availability_compatibility': calculate_availability_compatibility(
                mentor['availability_hours'], mentee['availability_hours']
            ),
            'style_match': style_match(mentor['mentorship_style'], mentee['preferred_style']),
            'timezone_match': timezone_match(mentor['timezone'], mentee['timezone']),
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
        
        # Predict
        feature_vector = pd.DataFrame([features])[feature_cols]
        success_prob = model.predict_proba(feature_vector)[0][1]
        
        recommendations.append({
            'mentor_id': mentor['mentor_id'],
            'mentor_name': mentor['name'],
            'match_score': success_prob,
            'domains': ', '.join(mentor['domains']) if mentor['domains'] else 'N/A',
            'experience_years': mentor['experience_years'],
            'rating': mentor['rating'],
            'style': mentor['mentorship_style'],
            'availability': mentor['availability_hours'],
            'domain_overlap': features['domain_overlap'],
            'style_match': '✓' if features['style_match'] == 1 else '✗',
            'timezone_match': '✓' if features['timezone_match'] == 1 else '✗'
        })
    
    # Sort and return top K
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values('match_score', ascending=False).head(top_k)
    
    return recommendations_df

def display_recommendations(recommendations):
    """Display recommendations in a nice format"""
    print(f"\n{'='*80}")
    print(f"TOP {len(recommendations)} MENTOR RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    for idx, rec in recommendations.iterrows():
        print(f"#{idx+1} - {rec['mentor_name']} (Match Score: {rec['match_score']:.1%})")
        print(f"     Domains: {rec['domains']}")
        print(f"     Experience: {rec['experience_years']} years | Rating: {rec['rating']:.1f}/5.0")
        print(f"     Style: {rec['style']} | Availability: {rec['availability']} hrs/month")
        print(f"     Domain Overlap: {rec['domain_overlap']:.0%} | Style Match: {rec['style_match']} | Timezone: {rec['timezone_match']}")
        print()

def main():
    """Main execution"""
    print("="*80)
    print(" "*20 + "MENTOR RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Load model and data
    model, feature_cols, mentors_df, mentees_df = load_model_and_data()
    
    # Interactive mode
    while True:
        print("\n" + "="*80)
        print("Options:")
        print("  1. Get recommendations for a specific mentee ID")
        print("  2. List all mentees")
        print("  3. Get recommendations for random mentee")
        print("  4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            mentee_id = input("Enter mentee ID: ").strip()
            try:
                mentee_id = int(mentee_id)
                recommendations = recommend_mentors(
                    mentee_id, model, feature_cols, mentors_df, mentees_df, top_k=5
                )
                if recommendations is not None:
                    display_recommendations(recommendations)
            except ValueError:
                print("❌ Invalid mentee ID. Please enter a number.")
        
        elif choice == '2':
            print("\n" + "="*80)
            print("AVAILABLE MENTEES")
            print("="*80)
            mentee_list = mentees_df[['mentee_id', 'name', 'current_level', 'desired_domains']].head(20)
            print(mentee_list.to_string(index=False))
            
            if len(mentees_df) > 20:
                print(f"\n... and {len(mentees_df) - 20} more")
        
        elif choice == '3':
            random_mentee = mentees_df.sample(1).iloc[0]
            mentee_id = random_mentee['mentee_id']
            
            recommendations = recommend_mentors(
                mentee_id, model, feature_cols, mentors_df, mentees_df, top_k=5
            )
            if recommendations is not None:
                display_recommendations(recommendations)
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()

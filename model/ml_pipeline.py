"""
Mentor-Mentee Matching: ML Pipeline Script
Complete pipeline from data loading to model training and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL not set. Add it to .env or environment variables before running the pipeline.")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering functions
def calculate_domain_overlap(mentor_domains, mentee_domains):
    """Calculate overlap between mentor expertise and mentee interests"""
    if not mentor_domains or not mentee_domains:
        return 0
    overlap = len(set(mentor_domains) & set(mentee_domains))
    return overlap / len(mentee_domains) if mentee_domains else 0

def calculate_skill_overlap(mentor_skills, mentee_skills):
    """Calculate skill overlap"""
    if not mentor_skills or not mentee_skills:
        return 0
    overlap = len(set(mentor_skills) & set(mentee_skills))
    total = len(set(mentor_skills) | set(mentee_skills))
    return overlap / total if total > 0 else 0

def calculate_availability_compatibility(mentor_hours, mentee_hours):
    """Calculate availability compatibility (0-1 scale)"""
    diff = abs(mentor_hours - mentee_hours)
    return max(0, 1 - (diff / 30))

def style_match(mentor_style, mentee_style):
    """Binary: do styles match?"""
    return 1 if mentor_style == mentee_style else 0

def timezone_match(mentor_tz, mentee_tz):
    """Binary: are they in same timezone?"""
    return 1 if mentor_tz == mentee_tz else 0

def industry_match(mentor_industry, mentee_industry):
    """Binary: same industry?"""
    return 1 if mentor_industry == mentee_industry else 0

def experience_level_compatibility(experience_years, current_level):
    """Calculate if mentor experience matches mentee level"""
    level_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    mentee_level_numeric = level_map.get(current_level, 2)
    
    if mentee_level_numeric == 1 and experience_years >= 10:
        return 1.0
    elif mentee_level_numeric == 2 and experience_years >= 7:
        return 0.8
    elif mentee_level_numeric == 3 and experience_years >= 5:
        return 0.6
    return 0.3

def load_data(db_url):
    """Load data from PostgreSQL database"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Load mentors
        mentors_df = pd.read_sql_query(text("SELECT * FROM mentors WHERE active = TRUE"), conn)
        print(f"✓ Loaded {len(mentors_df)} active mentors")
        
        # Load mentees
        mentees_df = pd.read_sql_query(text("SELECT * FROM mentees WHERE active = TRUE"), conn)
        print(f"✓ Loaded {len(mentees_df)} active mentees")
        
        # Load interactions
        interactions_df = pd.read_sql_query(text("""
            SELECT * FROM interactions 
            WHERE status IN ('completed', 'cancelled')
            AND mentor_accepted IS NOT NULL 
            AND mentee_accepted IS NOT NULL
        """), conn)
        print(f"✓ Loaded {len(interactions_df)} interactions")
    
    return mentors_df, mentees_df, interactions_df

def create_features(interactions_df, mentors_df, mentees_df):
    """Create feature vectors for all interactions"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    features_list = []
    
    for idx, row in interactions_df.iterrows():
        mentor_id = row['mentor_id']
        mentee_id = row['mentee_id']
        
        try:
            mentor = mentors_df[mentors_df['mentor_id'] == mentor_id].iloc[0]
            mentee = mentees_df[mentees_df['mentee_id'] == mentee_id].iloc[0]
            
            features = {
                'mentor_id': mentor_id,
                'mentee_id': mentee_id,
                'interaction_id': row['interaction_id'],
                
                # Similarity features
                'domain_overlap': calculate_domain_overlap(mentor['domains'], mentee['desired_domains']),
                'skill_overlap': calculate_skill_overlap(mentor['skills'], mentee.get('current_skills', [])),
                'availability_compatibility': calculate_availability_compatibility(
                    mentor['availability_hours'], mentee['availability_hours']
                ),
                
                # Categorical matches
                'style_match': style_match(mentor['mentorship_style'], mentee['preferred_style']),
                'timezone_match': timezone_match(mentor['timezone'], mentee['timezone']),
                'industry_match': industry_match(mentor['industry'], mentee['industry']),
                
                # Mentor characteristics
                'mentor_experience_years': mentor['experience_years'],
                'mentor_rating': mentor['rating'],
                'mentor_acceptance_rate': mentor['acceptance_rate'],
                'mentor_total_mentees': mentor['total_mentees'],
                
                # Mentee characteristics
                'mentee_level_numeric': {'beginner': 1, 'intermediate': 2, 'advanced': 3}.get(
                    mentee['current_level'], 2
                ),
                
                # Compatibility
                'experience_compatibility': experience_level_compatibility(
                    mentor['experience_years'], mentee['current_level']
                ),
                
                # Count features
                'mentor_domain_count': len(mentor['domains']) if mentor['domains'] else 0,
                'mentee_domain_count': len(mentee['desired_domains']) if mentee['desired_domains'] else 0,
                'mentor_skill_count': len(mentor['skills']) if mentor['skills'] else 0,
                
                # Target
                'successful_match': row['successful_match']
            }
            
            features_list.append(features)
            
        except Exception as e:
            print(f"⚠ Error processing interaction {row['interaction_id']}: {e}")
            continue
    
    features_df = pd.DataFrame(features_list)
    print(f"✓ Created {len(features_df)} feature vectors")
    
    return features_df

def analyze_features(features_df):
    """Analyze feature correlations with success"""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    feature_cols = [col for col in features_df.columns 
                    if col not in ['mentor_id', 'mentee_id', 'interaction_id', 'successful_match']]
    
    correlations = {}
    for col in feature_cols:
        corr = features_df[col].corr(features_df['successful_match'].astype(float))
        correlations[col] = corr
    
    correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    print("\nTop 10 Features by Correlation with Success:")
    for i, (feature, corr) in enumerate(list(correlations.items())[:10], 1):
        print(f"{i:2d}. {feature:35s}: {corr:+.4f}")
    
    return feature_cols

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    results = {}
    
    # Logistic Regression
    print("\n1. Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'predictions': y_pred_lr,
        'probabilities': y_pred_proba_lr,
        'scaler': scaler
    }
    
    print("✓ Logistic Regression trained")
    
    # Random Forest
    print("\n2. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf_model,
        'predictions': y_pred_rf,
        'probabilities': y_pred_proba_rf
    }
    
    print("✓ Random Forest trained")
    
    # Try XGBoost if available
    try:
        import xgboost as xgb
        
        print("\n3. Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': y_pred_xgb,
            'probabilities': y_pred_proba_xgb
        }
        
        print("✓ XGBoost trained")
        
    except ImportError:
        print("\n⚠ XGBoost not available (install with: pip install xgboost)")
    
    return results

def evaluate_models(results, y_test):
    """Evaluate and compare all models"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    comparison = []
    
    for model_name, result in results.items():
        y_pred = result['predictions']
        y_pred_proba = result['probabilities']
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        comparison.append(metrics)
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1 Score:  {metrics['F1']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
    print(f"\n🏆 Best Model: {best_model_name}")
    
    return comparison_df, best_model_name

def save_model(model_name, results, feature_cols):
    """Save the best model to disk"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model_data = results[model_name]
    
    joblib.dump(model_data['model'], 'mentor_matching_model.pkl')
    joblib.dump(feature_cols, 'feature_columns.pkl')
    
    if 'scaler' in model_data:
        joblib.dump(model_data['scaler'], 'feature_scaler.pkl')
        print("✓ Saved feature_scaler.pkl")
    
    print("✓ Saved mentor_matching_model.pkl")
    print("✓ Saved feature_columns.pkl")
    
    print(f"\nModel Type: {model_name}")
    print("Ready for deployment!")

def main():
    """Main pipeline execution"""
    print("\n" + "="*80)
    print(" "*20 + "MENTOR-MENTEE MATCHING ML PIPELINE")
    print("="*80)
    
    # Load data
    mentors_df, mentees_df, interactions_df = load_data(DB_URL)
    
    # Create features
    features_df = create_features(interactions_df, mentors_df, mentees_df)
    
    # Analyze features
    feature_cols = analyze_features(features_df)
    
    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    X = features_df[feature_cols]
    y = features_df['successful_match'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Success rate (train): {y_train.mean():.2%}")
    print(f"Success rate (test): {y_test.mean():.2%}")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    comparison_df, best_model_name = evaluate_models(results, y_test)
    
    # Save best model
    save_model(best_model_name, results, feature_cols)
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review model performance metrics above")
    print("  2. Use the saved model for predictions")
    print("  3. Create an API endpoint for production")
    print("  4. Implement feedback loop for continuous improvement")

if __name__ == "__main__":
    main()

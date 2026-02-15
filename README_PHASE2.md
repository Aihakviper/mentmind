# Phase 2: ML Model Development

Complete guide for building, training, and deploying your mentor-mentee matching ML model.

## 📋 Prerequisites

You should have completed Phase 1:
- ✅ PostgreSQL database set up
- ✅ Tables created (mentors, mentees, interactions)
- ✅ Synthetic data generated

## 🚀 Quick Start

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn sqlalchemy jupyter xgboost joblib
```

### Option 1: Interactive Notebook (Recommended for Learning)

```bash
jupyter notebook mentor_matching_ml.ipynb
```

The notebook walks through:
1. Data loading and exploration
2. Feature engineering
3. Model training (Logistic Regression, Random Forest, XGBoost)
4. Evaluation and comparison
5. Making predictions

### Option 2: Run Complete Pipeline Script

```bash
# Update DB_URL password in ml_pipeline.py first
python ml_pipeline.py
```

This will:
- Load data from PostgreSQL
- Engineer features automatically
- Train multiple models
- Compare performance
- Save the best model

Expected output:
```
Training set: 320 samples
Test set: 80 samples

MODEL COMPARISON
Model                 Accuracy  Precision  Recall    F1      ROC-AUC
Logistic Regression   0.7500    0.7200     0.7800    0.7500  0.8100
Random Forest         0.8250    0.8100     0.8400    0.8250  0.8750
XGBoost              0.8375    0.8250     0.8500    0.8375  0.8850

🏆 Best Model: XGBoost
```

### Option 3: Make Predictions with Trained Model

```bash
python inference.py
```

Interactive menu to:
- Get recommendations for specific mentees
- View all mentees
- Test with random mentees

## 🎯 What Gets Built

### Feature Engineering

The system creates these features to predict match success:

**Similarity Features:**
- `domain_overlap` - How much mentor expertise overlaps with mentee interests (0-1)
- `skill_overlap` - Jaccard similarity of skills (0-1)
- `availability_compatibility` - How well schedules align (0-1)

**Categorical Matches:**
- `style_match` - Do mentorship styles match? (binary)
- `timezone_match` - Same timezone? (binary)
- `industry_match` - Same industry? (binary)

**Profile Features:**
- Mentor: experience_years, rating, acceptance_rate, total_mentees
- Mentee: experience level (numeric)
- Compatibility: experience_level_compatibility score

**Count Features:**
- Number of domains, skills for both mentor/mentee

### Models Trained

1. **Logistic Regression** (Baseline)
   - Fast, interpretable
   - Good for understanding feature importance
   - Typically achieves ~75-80% accuracy

2. **Random Forest**
   - Handles non-linear relationships
   - Feature importance built-in
   - Typically achieves ~80-85% accuracy

3. **XGBoost** (Optional)
   - Best performance
   - Gradient boosting
   - Typically achieves ~83-88% accuracy

### Evaluation Metrics

The system evaluates models using:
- **Accuracy** - Overall correctness
- **Precision** - Of predicted successes, how many were correct?
- **Recall** - Of actual successes, how many did we find?
- **F1 Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve (best overall metric)

## 📊 Understanding Your Results

### Good Results Look Like:
```
ROC-AUC: 0.85+        # Excellent discrimination
Precision: 0.80+      # Most recommendations are good
Recall: 0.75+         # Finding most successful matches
```

### If Results Are Poor:
- Check data quality (enough interactions?)
- Review feature correlations (are features predictive?)
- Try hyperparameter tuning
- Collect more training data

## 🔍 Feature Importance Analysis

After training, check which features matter most:

```python
# In notebook or after running ml_pipeline.py
import joblib
model = joblib.load('mentor_matching_model.pkl')

# For Random Forest/XGBoost
feature_importance = model.feature_importances_
```

Typically important features:
1. `domain_overlap` - Most critical!
2. `mentor_rating` - Highly predictive
3. `availability_compatibility` - Important for engagement
4. `experience_compatibility` - Matters for learning

## 📁 Output Files

After training, you'll have:

```
├── mentor_matching_model.pkl    # Trained model (Random Forest or XGBoost)
├── feature_columns.pkl          # List of feature names
└── feature_scaler.pkl          # StandardScaler (if using Logistic Regression)
```

These files are used by `inference.py` for predictions.

## 🎓 Advanced Topics

### 1. Hyperparameter Tuning

Add to your script:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### 2. Cross-Validation

For more robust evaluation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X_train, y_train,
    cv=5, scoring='roc_auc'
)

print(f"Mean ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### 3. Text Embeddings (Advanced Feature)

Add semantic similarity from bios:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode bios
mentor_embeddings = model.encode(mentors_df['bio'].tolist())
mentee_embeddings = model.encode(mentees_df['goals'].tolist())

# Calculate cosine similarity for each pair
from sklearn.metrics.pairwise import cosine_similarity
bio_similarity = cosine_similarity(
    mentee_embedding.reshape(1, -1),
    mentor_embedding.reshape(1, -1)
)[0][0]
```

### 4. Ranking Model (Learning-to-Rank)

For better top-K recommendations:

```python
import xgboost as xgb

# Prepare data for ranking
dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'max_depth': 6
}

ranker = xgb.train(params, dtrain, num_boost_round=100)
```

## 🔄 Continuous Improvement

### Feedback Loop

Collect real user feedback to retrain:

```sql
-- Track which recommendations were accepted
CREATE TABLE recommendation_feedback (
    feedback_id SERIAL PRIMARY KEY,
    mentee_id INT,
    mentor_id INT,
    recommended_rank INT,  -- Was this the #1, #2, etc. recommendation?
    was_accepted BOOLEAN,
    feedback_date TIMESTAMP
);

-- Use this data to retrain your model monthly
```

### A/B Testing

Test new models against production:

```python
# Serve 90% traffic with model A, 10% with model B
import random

if random.random() < 0.9:
    recommendations = model_a.predict(features)
else:
    recommendations = model_b.predict(features)
    # Log this for analysis
```

## 🐛 Troubleshooting

### "Not enough data" error
- Need at least 50-100 interactions for training
- Run `generate_synthetic_data.py` again with more interactions

### Poor model performance (< 0.70 AUC)
- Check feature correlations - are they predictive?
- Verify data quality - realistic patterns?
- Try collecting more diverse training examples

### Model overfitting (train AUC >> test AUC)
- Reduce model complexity (lower max_depth)
- Add regularization
- Get more training data

### Features have NaN values
- Check for missing data in database
- Add imputation: `df.fillna(0)` or `df.fillna(df.mean())`

## 📈 Next Steps

After successful model training:

1. **Deploy to Production**
   - Create REST API with Flask/FastAPI
   - Host on cloud (AWS, GCP, Heroku)
   - Add authentication and rate limiting

2. **Monitor Performance**
   - Track recommendation acceptance rate
   - Log prediction latency
   - Set up alerts for model drift

3. **Enhance Features**
   - Add text embeddings from bios
   - Include user behavior data (clicks, time on profile)
   - Geographic distance calculations

4. **Scale**
   - Batch predictions for all users nightly
   - Cache recommendations
   - Use approximate nearest neighbors for speed

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Recommendation Systems Guide](https://developers.google.com/machine-learning/recommendation)
- [Collaborative Filtering Tutorial](https://realpython.com/build-recommendation-engine-collaborative-filtering/)

## 💡 Tips for Success

1. **Start Simple** - Get logistic regression working first
2. **Understand Features** - Plot correlations, check distributions
3. **Validate Thoroughly** - Use cross-validation, not just train/test split
4. **Iterate Quickly** - Don't over-engineer initially
5. **Measure What Matters** - Focus on business metrics (user satisfaction)

---

**Ready to build your matching engine!** 🚀

Need help? Common issues:
- Model not saving → Check file permissions
- Low accuracy → Review feature engineering
- Slow training → Reduce data size or model complexity

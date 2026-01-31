# Troubleshooting Guide

Common issues and solutions for the mentor-mentee matching system.

## Database Connection Issues

### Error: "AttributeError: 'OptionEngine' object has no attribute 'execute'"

**Cause:** SQLAlchemy version incompatibility with pandas

**Solution:** This has been fixed in the updated files. Make sure you're using the latest versions with `text()` wrapper:

```python
from sqlalchemy import create_engine, text

# Correct way (updated)
with engine.connect() as conn:
    df = pd.read_sql_query(text("SELECT * FROM table"), conn)

# Old way (causes error)
df = pd.read_sql_query("SELECT * FROM table", engine)
```

**Alternative:** Downgrade SQLAlchemy if you prefer:
```bash
pip install sqlalchemy==1.4.49
```

### Error: "permission denied for table X"

**Cause:** User doesn't have permissions on tables

**Solution:** Already fixed! You granted permissions in pgAdmin. If it happens again:

```sql
-- In psql or pgAdmin as postgres user
GRANT ALL PRIVILEGES ON TABLE mentors TO aihak;
GRANT ALL PRIVILEGES ON TABLE mentees TO aihak;
GRANT ALL PRIVILEGES ON TABLE interactions TO aihak;
GRANT ALL PRIVILEGES ON SEQUENCE mentors_mentor_id_seq TO aihak;
GRANT ALL PRIVILEGES ON SEQUENCE mentees_mentee_id_seq TO aihak;
GRANT ALL PRIVILEGES ON SEQUENCE interactions_interaction_id_seq TO aihak;
```

## Data Generation Issues

### Error: "ModuleNotFoundError: No module named 'faker'"

**Solution:**
```bash
pip install faker
```

### Synthetic data looks unrealistic

**Solution:** The data generator includes realistic patterns. Check:
- Domain overlap → success correlation
- Adjust `NUM_MENTORS`, `NUM_MENTEES`, `NUM_INTERACTIONS` in script
- Modify `calculate_match_quality()` function for different patterns

## ML Model Issues

### Error: "Not enough data to train model"

**Cause:** Too few interactions in database

**Solution:**
```python
# In generate_synthetic_data.py, increase:
NUM_INTERACTIONS = 500  # or higher
```

Then regenerate data:
```bash
python generate_synthetic_data.py
```

### Low model accuracy (< 0.70 ROC-AUC)

**Possible causes & solutions:**

1. **Not enough training data**
   ```bash
   # Generate more interactions
   python generate_synthetic_data.py
   ```

2. **Features not predictive**
   ```python
   # In notebook, check feature correlations:
   features_df.corr()['successful_match'].sort_values()
   ```

3. **Imbalanced classes**
   ```python
   # Check class distribution
   print(y_train.value_counts())
   
   # Use class weights in model
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(class_weight='balanced')
   ```

### Model overfitting (train accuracy >> test accuracy)

**Solutions:**
```python
# Reduce model complexity
RandomForestClassifier(max_depth=5, min_samples_split=10)

# Add regularization for Logistic Regression
LogisticRegression(C=0.1)

# Use cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

## API Issues

### Error: "Address already in use"

**Cause:** Port 5000 is already taken

**Solution:**
```python
# In api.py, change port:
app.run(debug=True, host='0.0.0.0', port=5001)
```

Or kill the existing process:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Error: "Model files not found"

**Cause:** Haven't trained model yet

**Solution:**
```bash
# Train the model first
python ml_pipeline.py

# This creates:
# - mentor_matching_model.pkl
# - feature_columns.pkl
# - feature_scaler.pkl (if using Logistic Regression)
```

### CORS errors in browser

**Cause:** Frontend on different port/domain

**Solution:** Already handled by `flask-cors`. If issues persist:
```python
# In api.py, configure CORS more specifically:
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://yourdomain.com"]
    }
})
```

## Jupyter Notebook Issues

### Kernel keeps dying

**Causes & solutions:**

1. **Out of memory**
   ```python
   # Reduce data size
   mentors_df = mentors_df.head(100)
   mentees_df = mentees_df.head(150)
   ```

2. **Infinite loop in feature engineering**
   - Check your loops have proper termination
   - Add print statements to debug

### Can't connect to database from notebook

**Solution:**
```python
# Test connection first
from sqlalchemy import create_engine, text

DB_URL = "postgresql://aihak:your_password@localhost:5432/mentor_ai"
engine = create_engine(DB_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("✓ Connection successful!")
```

## Package Installation Issues

### Error: "Microsoft Visual C++ 14.0 or greater is required"

**Cause:** Some packages need C++ compiler on Windows

**Solution:**
1. Install Visual Studio Build Tools
2. Or use pre-built wheels:
   ```bash
   pip install --only-binary :all: package_name
   ```

### Error: "No matching distribution found"

**Cause:** Package not compatible with Python version

**Solution:**
```bash
# Check Python version
python --version

# Update pip
python -m pip install --upgrade pip

# Install with specific version
pip install package_name==version
```

## Performance Issues

### Training is very slow

**Solutions:**

1. **Use fewer estimators**
   ```python
   RandomForestClassifier(n_estimators=50)  # instead of 100
   ```

2. **Reduce data size for testing**
   ```python
   # Sample data
   features_df = features_df.sample(frac=0.5)
   ```

3. **Use all CPU cores**
   ```python
   RandomForestClassifier(n_jobs=-1)
   ```

### API is slow

**Solutions:**

1. **Cache predictions**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_cached_recommendations(mentee_id):
       # ... recommendation logic
   ```

2. **Batch process recommendations**
   ```python
   # Pre-compute for all users nightly
   # Store in database
   ```

3. **Use simpler model**
   ```python
   # Logistic Regression is faster than Random Forest
   ```

## Data Quality Issues

### Arrays showing as strings in database

**Cause:** PostgreSQL array not properly parsed

**Solution:**
```python
# When reading from DB
import ast

# Convert string to list
df['skills'] = df['skills'].apply(ast.literal_eval)
```

### Missing values causing errors

**Solution:**
```python
# Check for missing values
print(df.isnull().sum())

# Fill missing values
df['rating'].fillna(df['rating'].mean(), inplace=True)
df['domains'].fillna([], inplace=True)
```

## Still Having Issues?

### Debug checklist:

1. ✅ PostgreSQL is running
2. ✅ Database credentials are correct
3. ✅ Tables exist and have data
4. ✅ All packages installed
5. ✅ Python version is 3.8+
6. ✅ Virtual environment activated (if using one)

### Get detailed error info:

```python
import traceback

try:
    # Your code here
    pass
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

### Common package versions that work well:

```bash
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 sqlalchemy==2.0.20 psycopg2-binary==2.9.7
```

---

**If you're still stuck:**
1. Check the error message carefully
2. Google the specific error
3. Check package documentation
4. Try a minimal example to isolate the issue

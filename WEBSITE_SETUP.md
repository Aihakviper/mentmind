# MentMinds Website - Complete Setup Guide

## 📁 Files Overview

### Website Files
1. **mentminds_website.html** - Main landing page
2. **mentor_matching.html** - AI-powered mentor matching interface

### Backend Files (Already Created)
3. **api.py** - Flask API for mentor recommendations
4. **ml_pipeline_nigeria.py** - ML model training
5. **inference.py** - CLI for testing predictions
6. **generate_synthetic_data.py** - Data generation

### Database Files
7. **schema.sql** - Database schema
8. **setup_schema.py** - Schema setup script

## 🚀 Complete Setup Instructions

### Step 1: Database Setup

```bash
# 1. Make sure PostgreSQL is running
# 2. Create database (if not already done)
psql -U postgres -c "CREATE DATABASE mentor_ai;"

# 3. Grant permissions
psql -U postgres -d mentor_ai
```

```sql
GRANT ALL PRIVILEGES ON DATABASE mentor_ai TO aihak;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aihak;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aihak;
\q
```

```bash
# 4. Create tables
python setup_schema.py

# 5. Generate data
python generate_synthetic_data.py
```

### Step 2: Train ML Model

```bash
# Train the matching model
python ml_pipeline_nigeria.py
```

This creates:
- `mentor_matching_model.pkl`
- `feature_columns.pkl`
- `feature_scaler.pkl` (if using Logistic Regression)

### Step 3: Start the API

```bash
# Update password in api.py first (line 35)
# Then run:
python api.py
```

API will start at: `http://localhost:5000`

Endpoints available:
- `GET /api/health` - Health check
- `GET /api/mentors` - List all mentors
- `GET /api/mentees` - List all mentees
- `POST /api/recommend` - Get mentor recommendations
- `POST /api/match-score` - Get match score for a pair

### Step 4: Open the Website

```bash
# Open in browser:
# - mentminds_website.html (main site)
# - mentor_matching.html (matching interface)

# Or use a local server:
python -m http.server 8000

# Then visit:
# http://localhost:8000/mentminds_website.html
# http://localhost:8000/mentor_matching.html
```

## 🌐 Website Structure

### Main Website (mentminds_website.html)

**Home Page:**
- Hero section with stats
- About MentMinds story
- How it works
- CTA to find mentors

**For Mentors Page:**
- Benefits of mentoring
- Impact you'll make
- Apply to mentor

**For Mentees Page:**
- What you'll gain
- Personalized guidance
- Browse mentors button → matching page

### Matching Interface (mentor_matching.html)

**Left Sidebar:**
- Profile form
  - Name, experience level
  - Domains to learn
  - Goals, industry
  - Learning style
  - Availability

**Main Area:**
- Recommended mentors (AI-powered)
- Match scores and details
- Connect & chat buttons

**Chat Modal:**
- Real-time messaging interface
- Send/receive messages

## 🔗 Navigation Flow

```
mentminds_website.html
    ├── Home Page
    ├── About Section
    ├── For Mentors Page
    ├── For Mentees Page
    └── "Find a Mentor" button → mentor_matching.html

mentor_matching.html
    ├── Profile Form (sidebar)
    ├── AI Recommendations (main)
    ├── Chat Modal
    └── "Back to Home" → mentminds_website.html
```

## 🎨 Design Features

### Color Palette
- Primary: `#E67E22` (Orange)
- Secondary: `#27AE60` (Green)
- Accent: `#F39C12` (Yellow-Orange)
- Background: `#FDF8F3` (Warm Light)

### Typography
- Headers: **Fraunces** (Serif)
- Body: **DM Sans** (Sans-serif)

### Animations
- Smooth page transitions
- Hover effects on cards
- Loading states
- Modal slide-ups

## ⚙️ Configuration

### API Connection

In `mentor_matching.html`, line 654:
```javascript
const API_URL = 'http://localhost:5000/api';
```

If your API runs on a different port, update this.

### API CORS

In `api.py`, CORS is enabled for all origins:
```python
from flask_cors import CORS
CORS(app)
```

For production, restrict to your domain:
```python
CORS(app, resources={
    r"/api/*": {"origins": ["https://yourdomain.com"]}
})
```

## 🧪 Testing the Complete System

### 1. Test API
```bash
# Terminal 1: Start API
python api.py

# Terminal 2: Test endpoints
curl http://localhost:5000/api/health
curl http://localhost:5000/api/mentors
```

### 2. Test Website
```bash
# Open mentminds_website.html in browser
# Click "Find a Mentor"
# Fill out profile
# Click "Find My Mentors"
```

You should see:
- AI-recommended mentors
- Match scores (e.g., 87% match)
- Domain overlap details
- Connect buttons

### 3. Test Chat
```bash
# Click "Connect" on any mentor
# Type a message
# See simulated response
```

## 📊 Data Flow

```
User fills form
    ↓
mentor_matching.html collects data
    ↓
POST /api/recommend
    ↓
api.py processes request
    ↓
ml_model predicts matches
    ↓
Returns top 5 mentors with scores
    ↓
Display in UI with details
    ↓
User clicks "Connect"
    ↓
Chat modal opens
```

## 🚀 Deployment to Production

### Option 1: Simple Hosting (Static + API)

**Frontend (Netlify/Vercel):**
```bash
# Upload these files:
- mentminds_website.html
- mentor_matching.html
```

**Backend (Heroku/Railway):**
```bash
# Create Procfile
echo "web: python api.py" > Procfile

# Create requirements.txt
pip freeze > requirements.txt

# Deploy
git init
git add .
git commit -m "Initial commit"
heroku create mentminds-api
git push heroku main
```

### Option 2: All-in-One (VPS)

**On DigitalOcean/AWS/Linode:**
```bash
# 1. Install dependencies
sudo apt update
sudo apt install python3 postgresql nginx

# 2. Setup PostgreSQL
sudo -u postgres createdb mentor_ai
sudo -u postgres psql -d mentor_ai < schema.sql

# 3. Run data generation
python generate_synthetic_data.py
python ml_pipeline_nigeria.py

# 4. Setup nginx
sudo nano /etc/nginx/sites-available/mentminds

# Add:
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        root /var/www/mentminds;
        try_files $uri $uri/ =404;
    }
    
    location /api {
        proxy_pass http://localhost:5000;
    }
}

# 5. Start API with systemd
sudo nano /etc/systemd/system/mentminds-api.service

# Add service configuration
sudo systemctl enable mentminds-api
sudo systemctl start mentminds-api
```

## 🔐 Security Checklist

Before going live:
- [ ] Add user authentication
- [ ] Sanitize all inputs
- [ ] Add rate limiting to API
- [ ] Use HTTPS (SSL certificate)
- [ ] Secure database credentials
- [ ] Add CORS restrictions
- [ ] Implement session management
- [ ] Add input validation

## 📱 Mobile Responsiveness

Both pages are fully responsive:
- Desktop: 1200px+ (full sidebar + main)
- Tablet: 768px-1199px (stacked layout)
- Mobile: <768px (single column)

## 🐛 Troubleshooting

### API Not Connecting
```javascript
// Check console for errors
// Update API_URL in mentor_matching.html
// Make sure api.py is running
// Check firewall/CORS settings
```

### No Mentors Showing
```sql
-- Check database has data
SELECT COUNT(*) FROM mentors;
SELECT COUNT(*) FROM mentees;

-- Check API is working
curl http://localhost:5000/api/mentors
```

### Chat Not Working
```javascript
// Chat is currently simulated
// For real-time chat, implement:
// - WebSockets (Socket.IO)
// - Or long polling
// - Or Firebase Realtime Database
```

## 📈 Next Steps

### Phase 1 (MVP) - ✅ DONE
- [x] Landing page
- [x] Mentor matching interface
- [x] ML-powered recommendations
- [x] Chat UI

### Phase 2 (Enhancement)
- [ ] User authentication (login/signup)
- [ ] Save mentee profiles to database
- [ ] Real-time chat (WebSockets)
- [ ] Mentor profiles page
- [ ] Booking system for sessions

### Phase 3 (Growth)
- [ ] Progress tracking dashboard
- [ ] Resource sharing
- [ ] Success metrics
- [ ] Community forum
- [ ] Mobile app (React Native)

## 📞 Support

For issues or questions:
1. Check this guide
2. Review TROUBLESHOOTING.md
3. Check API logs
4. Test endpoints with curl/Postman

## 🎉 You're Ready!

Your complete MentMinds platform is set up:
- ✅ Beautiful landing page
- ✅ AI-powered matching
- ✅ Interactive chat interface
- ✅ ML model trained and running
- ✅ API serving recommendations

**Go launch and help young Nigerians find amazing mentors!** 🇳🇬🚀

"""
Synthetic Data Generator for Mentor-Mentee Matching System
Generates realistic data with patterns that ML models can learn from
"""

import random
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

fake = Faker()

# Configuration
NUM_MENTORS = 200
NUM_MENTEES = 300
NUM_INTERACTIONS = 400  # Some mentors/mentees will have multiple matches

# Domain and Skills Data
DOMAINS = [
    'Data Science', 'Machine Learning', 'Software Engineering', 'Web Development',
    'Mobile Development', 'DevOps', 'Cybersecurity', 'Cloud Computing',
    'Product Management', 'UX Design', 'Business Analytics', 'Leadership',
    'Career Development', 'Entrepreneurship', 'Marketing', 'Sales'
]

SKILLS = {
    'Data Science': ['Python', 'R', 'SQL', 'Statistics', 'Data Visualization', 'Pandas', 'NumPy'],
    'Machine Learning': ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Deep Learning', 'NLP', 'Computer Vision'],
    'Software Engineering': ['Python', 'Java', 'C++', 'JavaScript', 'Git', 'Algorithms', 'System Design'],
    'Web Development': ['JavaScript', 'React', 'Node.js', 'HTML/CSS', 'TypeScript', 'REST APIs', 'GraphQL'],
    'Mobile Development': ['Swift', 'Kotlin', 'React Native', 'Flutter', 'iOS', 'Android', 'Mobile UI/UX'],
    'DevOps': ['Docker', 'Kubernetes', 'CI/CD', 'AWS', 'Terraform', 'Linux', 'Monitoring'],
    'Cybersecurity': ['Network Security', 'Penetration Testing', 'Cryptography', 'Security Auditing', 'SIEM'],
    'Cloud Computing': ['AWS', 'Azure', 'GCP', 'Serverless', 'Cloud Architecture', 'Microservices'],
    'Product Management': ['Product Strategy', 'Roadmapping', 'User Research', 'Agile', 'Stakeholder Management'],
    'UX Design': ['Figma', 'User Research', 'Prototyping', 'Interaction Design', 'Usability Testing'],
    'Business Analytics': ['Excel', 'Tableau', 'Power BI', 'SQL', 'Business Intelligence', 'KPIs'],
    'Leadership': ['Team Management', 'Communication', 'Strategic Thinking', 'Coaching', 'Decision Making'],
    'Career Development': ['Resume Building', 'Interview Prep', 'Networking', 'Personal Branding', 'Career Planning'],
    'Entrepreneurship': ['Business Planning', 'Fundraising', 'MVP Development', 'Market Research', 'Pitching'],
    'Marketing': ['Digital Marketing', 'SEO', 'Content Marketing', 'Social Media', 'Analytics'],
    'Sales': ['B2B Sales', 'Negotiation', 'CRM', 'Lead Generation', 'Sales Strategy']
}

INDUSTRIES = [
    'Technology', 'Finance', 'Healthcare', 'Education', 'Consulting',
    'E-commerce', 'Media', 'Manufacturing', 'Retail', 'Telecommunications'
]

MENTORSHIP_STYLES = ['structured', 'flexible', 'hands-on', 'advisory', 'collaborative']
EXPERIENCE_LEVELS = ['beginner', 'intermediate', 'advanced']
LEARNING_PACES = ['fast', 'moderate', 'slow']

# Nigerian cities for location
NIGERIAN_CITIES = [
    'Lagos', 'Abuja', 'Kano', 'Ibadan', 'Port Harcourt',
    'Benin City', 'Kaduna', 'Enugu', 'Jos', 'Calabar',
    'Owerri', 'Warri', 'Abeokuta', 'Ilorin', 'Akure'
]

def generate_bio(domains, skills, role='mentor'):
    """Generate realistic bio text"""
    if role == 'mentor':
        templates = [
            f"Experienced professional in {', '.join(domains[:2])} with expertise in {', '.join(skills[:3])}. Passionate about helping others grow.",
            f"Senior {domains[0]} specialist. I love mentoring and sharing knowledge about {', '.join(skills[:2])}.",
            f"{random.choice(['10+', '15+', '8+'])} years in {domains[0]}. Focused on {', '.join(domains[:2])} and enjoy guiding early-career professionals.",
        ]
    else:
        templates = [
            f"Aspiring {domains[0]} professional looking to learn {', '.join(skills[:2])}. Eager to grow and develop my career.",
            f"Currently working in {domains[0]}, want to transition into {domains[1] if len(domains) > 1 else domains[0]}.",
            f"Motivated learner seeking guidance in {', '.join(domains[:2])}. Ready to put in the work!",
        ]
    return random.choice(templates)

def generate_mentor():
    """Generate a synthetic mentor with realistic attributes"""
    # Select 1-3 domains
    num_domains = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    mentor_domains = random.sample(DOMAINS, num_domains)
    
    # Generate skills based on domains
    mentor_skills = []
    for domain in mentor_domains:
        if domain in SKILLS:
            domain_skills = random.sample(SKILLS[domain], random.randint(2, 4))
            mentor_skills.extend(domain_skills)
    mentor_skills = list(set(mentor_skills))  # Remove duplicates
    
    # Experience correlates with number of domains and skills
    experience_years = random.randint(5, 20)
    
    # More experienced mentors tend to have higher ratings
    base_rating = 3.0 + (experience_years / 20) * 2  # 3.0 to 5.0
    rating = min(5.0, max(1.0, base_rating + random.gauss(0, 0.3)))
    
    # Acceptance rate somewhat correlates with availability
    availability_hours = random.choice([5, 10, 15, 20, 25, 30])
    base_acceptance = 0.5 + (availability_hours / 60)
    acceptance_rate = min(1.0, max(0.2, base_acceptance + random.gauss(0, 0.15)))
    
    mentor = {
        'name': fake.name(),
        'email': fake.email(),
        'skills': mentor_skills,
        'domains': mentor_domains,
        'experience_years': experience_years,
        'industry': random.choice(INDUSTRIES),
        'mentorship_style': random.choice(MENTORSHIP_STYLES),
        'availability_hours': availability_hours,
        'max_mentees': random.choice([2, 3, 4, 5]),
        'bio': generate_bio(mentor_domains, mentor_skills, 'mentor'),
        'location': random.choice(NIGERIAN_CITIES),
        'rating': round(rating, 2),
        'total_mentees': random.randint(0, 10),
        'acceptance_rate': round(acceptance_rate, 2),
        'active': random.choice([True] * 9 + [False])  # 90% active
    }
    
    return mentor

def generate_mentee():
    """Generate a synthetic mentee with realistic attributes"""
    # Select 1-2 desired domains
    num_domains = random.choices([1, 2], weights=[0.6, 0.4])[0]
    desired_domains = random.sample(DOMAINS, num_domains)
    
    # Current skills - fewer than what they want to learn
    current_skills = []
    for domain in desired_domains:
        if domain in SKILLS and random.random() > 0.3:  # 70% chance they have some skills
            domain_skills = random.sample(SKILLS[domain], random.randint(1, 2))
            current_skills.extend(domain_skills)
    
    # Experience level affects goals
    current_level = random.choice(EXPERIENCE_LEVELS)
    
    goals_templates = {
        'beginner': [
            f"Build foundational skills in {desired_domains[0]} and land my first role",
            f"Transition into {desired_domains[0]} from a different field",
            f"Learn the basics of {', '.join(desired_domains)} and build projects"
        ],
        'intermediate': [
            f"Advance my career in {desired_domains[0]} and take on more senior responsibilities",
            f"Deepen my expertise in {', '.join(desired_domains)}",
            f"Prepare for senior roles in {desired_domains[0]}"
        ],
        'advanced': [
            f"Transition into leadership roles in {desired_domains[0]}",
            f"Become an expert in {desired_domains[0]} and mentor others",
            f"Navigate career growth and strategic decisions in {', '.join(desired_domains)}"
        ]
    }
    
    mentee = {
        'name': fake.name(),
        'email': fake.email(),
        'current_skills': current_skills,
        'goals': random.choice(goals_templates[current_level]),
        'desired_domains': desired_domains,
        'current_level': current_level,
        'industry': random.choice(INDUSTRIES),
        'preferred_style': random.choice(MENTORSHIP_STYLES),
        'availability_hours': random.choice([5, 10, 15, 20]),
        'location': random.choice(NIGERIAN_CITIES),
        'learning_pace': random.choice(LEARNING_PACES),
        'active': random.choice([True] * 9 + [False])
    }
    
    return mentee

def calculate_match_quality(mentor, mentee):
    """
    Calculate match quality score (0-1) based on various factors
    This simulates what makes a "good match" in reality
    """
    score = 0.5  # Base score
    
    # Domain overlap (most important factor)
    domain_overlap = len(set(mentor['domains']) & set(mentee['desired_domains']))
    if domain_overlap > 0:
        score += 0.25 * (domain_overlap / len(mentee['desired_domains']))
    
    # Skill complementarity - mentor has skills mentee wants to learn
    mentee_desired_skills = []
    for domain in mentee['desired_domains']:
        if domain in SKILLS:
            mentee_desired_skills.extend(SKILLS[domain])
    
    skill_match = len(set(mentor['skills']) & set(mentee_desired_skills))
    if skill_match > 0:
        score += 0.15
    
    # Style match
    if mentor['mentorship_style'] == mentee['preferred_style']:
        score += 0.1
    
    # Availability compatibility
    availability_diff = abs(mentor['availability_hours'] - mentee['availability_hours'])
    if availability_diff <= 5:
        score += 0.1
    elif availability_diff <= 10:
        score += 0.05
    
    # Industry match can be helpful
    if mentor['industry'] == mentee['industry']:
        score += 0.05
    
    # Experience level - beginners benefit from more experienced mentors
    if mentee['current_level'] == 'beginner' and mentor['experience_years'] >= 10:
        score += 0.05
    
    # Add some randomness
    score += random.gauss(0, 0.1)
    
    return min(1.0, max(0.0, score))

def generate_interaction(mentor_id, mentee_id, mentor_data, mentee_data):
    """
    Generate a realistic interaction/match
    Success depends on match quality
    """
    match_quality = calculate_match_quality(mentor_data, mentee_data)
    
    # Higher match quality = higher acceptance rates
    mentor_accepts = random.random() < (0.5 + match_quality * 0.4)
    mentee_accepts = random.random() < (0.6 + match_quality * 0.3)
    
    if not (mentor_accepts and mentee_accepts):
        # Match was proposed but not accepted
        return {
            'mentor_id': mentor_id,
            'mentee_id': mentee_id,
            'match_date': fake.date_time_between(start_date='-2y', end_date='now'),
            'mentor_accepted': mentor_accepts,
            'mentee_accepted': mentee_accepts,
            'status': 'cancelled',
            'meetings_held': 0,
            'duration_weeks': None,
            'mentor_rating': None,
            'mentee_rating': None,
            'overall_rating': None,
            'successful_match': False,
            'feedback_mentor': None,
            'feedback_mentee': None,
            'ended_at': None
        }
    
    # Both accepted - create active or completed mentorship
    match_date = fake.date_time_between(start_date='-1y', end_date='-1m')
    
    # Duration influenced by match quality
    base_duration = 12  # weeks
    duration_weeks = int(base_duration * (0.5 + match_quality * 1.0) + random.randint(-4, 8))
    duration_weeks = max(2, min(52, duration_weeks))
    
    # Status based on duration
    is_completed = random.random() < 0.7  # 70% of matches are completed
    status = 'completed' if is_completed else 'active'
    
    # Meetings held based on duration and engagement
    expected_meetings = duration_weeks // 2  # Biweekly meetings
    meetings_held = int(expected_meetings * (0.7 + match_quality * 0.3)) + random.randint(-2, 2)
    meetings_held = max(1, meetings_held)
    
    # Ratings based on match quality
    base_mentor_rating = 2 + match_quality * 3  # 2-5 scale
    mentor_rating = int(round(base_mentor_rating + random.gauss(0, 0.5)))
    mentor_rating = max(1, min(5, mentor_rating))
    
    base_mentee_rating = 2 + match_quality * 3
    mentee_rating = int(round(base_mentee_rating + random.gauss(0, 0.5)))
    mentee_rating = max(1, min(5, mentee_rating))
    
    overall_rating = (mentor_rating + mentee_rating) / 2.0
    
    # Successful match if ratings are good
    successful_match = overall_rating >= 3.5 and meetings_held >= 3
    
    ended_at = match_date + timedelta(weeks=duration_weeks) if is_completed else None
    
    feedback_options = [
        "Great experience, learned a lot!",
        "Very helpful and supportive mentor.",
        "Good match, achieved my learning goals.",
        "Valuable insights and guidance.",
        "Could have been better aligned on expectations.",
        "Schedule conflicts made it challenging.",
        None, None  # Some matches don't have feedback
    ]
    
    interaction = {
        'mentor_id': mentor_id,
        'mentee_id': mentee_id,
        'match_date': match_date,
        'mentor_accepted': True,
        'mentee_accepted': True,
        'status': status,
        'meetings_held': meetings_held,
        'duration_weeks': duration_weeks,
        'mentor_rating': mentor_rating if is_completed else None,
        'mentee_rating': mentee_rating if is_completed else None,
        'overall_rating': round(overall_rating, 2) if is_completed else None,
        'successful_match': successful_match if is_completed else None,
        'feedback_mentor': random.choice(feedback_options) if is_completed else None,
        'feedback_mentee': random.choice(feedback_options) if is_completed else None,
        'ended_at': ended_at
    }
    
    return interaction

def insert_data_to_db(db_url):
    """Insert generated data into PostgreSQL database using SQLAlchemy"""
    
    print("Connecting to database...")
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Clear existing data
        print("Clearing existing data...")
        conn.execute(text("DELETE FROM interactions;"))
        conn.execute(text("DELETE FROM mentees;"))
        conn.execute(text("DELETE FROM mentors;"))
        conn.commit()
        
        # Generate and insert mentors
        print(f"Generating {NUM_MENTORS} mentors...")
        mentors = []
        mentor_ids = []
        
        for i in range(NUM_MENTORS):
            mentor = generate_mentor()
            mentors.append(mentor)
            
            result = conn.execute(text("""
                INSERT INTO mentors (name, email, skills, domains, experience_years, industry,
                                   mentorship_style, availability_hours, max_mentees, bio,
                                   location, rating, total_mentees, acceptance_rate, active)
                VALUES (:name, :email, :skills, :domains, :experience_years, :industry,
                       :mentorship_style, :availability_hours, :max_mentees, :bio,
                       :location, :rating, :total_mentees, :acceptance_rate, :active)
                RETURNING mentor_id
            """), {
                'name': mentor['name'],
                'email': mentor['email'],
                'skills': mentor['skills'],
                'domains': mentor['domains'],
                'experience_years': mentor['experience_years'],
                'industry': mentor['industry'],
                'mentorship_style': mentor['mentorship_style'],
                'availability_hours': mentor['availability_hours'],
                'max_mentees': mentor['max_mentees'],
                'bio': mentor['bio'],
                'location': mentor['location'],
                'rating': mentor['rating'],
                'total_mentees': mentor['total_mentees'],
                'acceptance_rate': mentor['acceptance_rate'],
                'active': mentor['active']
            })
            
            mentor_id = result.fetchone()[0]
            mentor_ids.append(mentor_id)
            
            if (i + 1) % 50 == 0:
                print(f"  Inserted {i + 1} mentors...")
        
        conn.commit()
        print(f"✓ Inserted {NUM_MENTORS} mentors")
        
        # Generate and insert mentees
        print(f"Generating {NUM_MENTEES} mentees...")
        mentees = []
        mentee_ids = []
        
        for i in range(NUM_MENTEES):
            mentee = generate_mentee()
            mentees.append(mentee)
            
            result = conn.execute(text("""
                INSERT INTO mentees (name, email, current_skills, goals, desired_domains,
                                   current_level, industry, preferred_style, availability_hours,
                                   location, learning_pace, active)
                VALUES (:name, :email, :current_skills, :goals, :desired_domains,
                       :current_level, :industry, :preferred_style, :availability_hours,
                       :location, :learning_pace, :active)
                RETURNING mentee_id
            """), {
                'name': mentee['name'],
                'email': mentee['email'],
                'current_skills': mentee['current_skills'],
                'goals': mentee['goals'],
                'desired_domains': mentee['desired_domains'],
                'current_level': mentee['current_level'],
                'industry': mentee['industry'],
                'preferred_style': mentee['preferred_style'],
                'availability_hours': mentee['availability_hours'],
                'location': mentee['location'],
                'learning_pace': mentee['learning_pace'],
                'active': mentee['active']
            })
            
            mentee_id = result.fetchone()[0]
            mentee_ids.append(mentee_id)
            
            if (i + 1) % 50 == 0:
                print(f"  Inserted {i + 1} mentees...")
        
        conn.commit()
        print(f" Inserted {NUM_MENTEES} mentees")
        
        # Generate and insert interactions
        print(f"Generating {NUM_INTERACTIONS} interactions...")
        used_pairs = set()
        
        for i in range(NUM_INTERACTIONS):
            # Select random mentor and mentee (avoid duplicates)
            attempts = 0
            while attempts < 100:
                mentor_idx = random.randint(0, NUM_MENTORS - 1)
                mentee_idx = random.randint(0, NUM_MENTEES - 1)
                pair = (mentor_ids[mentor_idx], mentee_ids[mentee_idx])
                
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    break
                attempts += 1
            
            if attempts >= 100:
                continue  # Skip if we can't find a unique pair
            
            interaction = generate_interaction(
                mentor_ids[mentor_idx],
                mentee_ids[mentee_idx],
                mentors[mentor_idx],
                mentees[mentee_idx]
            )
            
            conn.execute(text("""
                INSERT INTO interactions (mentor_id, mentee_id, match_date, mentor_accepted,
                                        mentee_accepted, status, meetings_held, duration_weeks,
                                        mentor_rating, mentee_rating, overall_rating,
                                        successful_match, feedback_mentor, feedback_mentee, ended_at)
                VALUES (:mentor_id, :mentee_id, :match_date, :mentor_accepted,
                       :mentee_accepted, :status, :meetings_held, :duration_weeks,
                       :mentor_rating, :mentee_rating, :overall_rating,
                       :successful_match, :feedback_mentor, :feedback_mentee, :ended_at)
            """), {
                'mentor_id': interaction['mentor_id'],
                'mentee_id': interaction['mentee_id'],
                'match_date': interaction['match_date'],
                'mentor_accepted': interaction['mentor_accepted'],
                'mentee_accepted': interaction['mentee_accepted'],
                'status': interaction['status'],
                'meetings_held': interaction['meetings_held'],
                'duration_weeks': interaction['duration_weeks'],
                'mentor_rating': interaction['mentor_rating'],
                'mentee_rating': interaction['mentee_rating'],
                'overall_rating': interaction['overall_rating'],
                'successful_match': interaction['successful_match'],
                'feedback_mentor': interaction['feedback_mentor'],
                'feedback_mentee': interaction['feedback_mentee'],
                'ended_at': interaction['ended_at']
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Inserted {i + 1} interactions...")
        
        conn.commit()
        print(f"✓ Inserted {NUM_INTERACTIONS} interactions")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATA GENERATION SUMMARY")
        print("="*60)
        
        result = conn.execute(text("SELECT COUNT(*), AVG(rating), AVG(experience_years) FROM mentors WHERE active = TRUE"))
        mentor_stats = result.fetchone()
        print(f"Active Mentors: {mentor_stats[0]}")
        print(f"  Avg Rating: {mentor_stats[1]:.2f}")
        print(f"  Avg Experience: {mentor_stats[2]:.1f} years")
        
        result = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT current_level) FROM mentees WHERE active = TRUE"))
        mentee_stats = result.fetchone()
        print(f"\nActive Mentees: {mentee_stats[0]}")
        
        result = conn.execute(text("""
            SELECT status, COUNT(*), AVG(overall_rating), AVG(meetings_held)
            FROM interactions
            WHERE status IN ('active', 'completed')
            GROUP BY status
        """))
        
        print("\nInteractions:")
        for row in result.fetchall():
            status, count, avg_rating, avg_meetings = row
            print(f"  {status.capitalize()}: {count} matches")
            if avg_rating:
                print(f"    Avg Rating: {avg_rating:.2f}")
            if avg_meetings:
                print(f"    Avg Meetings: {avg_meetings:.1f}")
        
        result = conn.execute(text("""
            SELECT COUNT(*), AVG(overall_rating)
            FROM interactions
            WHERE successful_match = TRUE
        """))
        success_stats = result.fetchone()
        print(f"\nSuccessful Matches: {success_stats[0]}")
        if success_stats[1]:
            print(f"  Avg Rating: {success_stats[1]:.2f}")
    
    print("\n" + "="*60)
    print(" Data generation complete!")
    print("="*60)

if __name__ == "__main__":
    # Database connection URL
   
    import os
    from dotenv import load_dotenv
    load_dotenv()
    DB_URL = os.getenv("DATABASE_URL")
    if not DB_URL:
        raise RuntimeError("DATABASE_URL not set. Add it to .env or environment variables before running data generator.")
    
    print("="*60)
    print("MENTOR-MENTEE MATCHING - SYNTHETIC DATA GENERATOR")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Mentors: {NUM_MENTORS}")
    print(f"  Mentees: {NUM_MENTEES}")
    print(f"  Interactions: {NUM_INTERACTIONS}")
    
    
    response = input("\nProceed with data generation? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        try:
            insert_data_to_db(DB_URL)
        except Exception as e:
            print(f"\n Error: {e}")
            print("\nMake sure:")
            print("  1. PostgreSQL is running")
            print("  2. Database credentials are correct in DB_URL")
            print("  3. Tables are created (run the SQL schema first)")
            print("  4. Required Python packages are installed:")
            print("     pip install sqlalchemy faker numpy")
    else:
        print("\nData generation cancelled.")
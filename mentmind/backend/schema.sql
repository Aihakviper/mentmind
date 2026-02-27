-- ============================================================
-- MentMinds Database Schema
-- Run this on your Render PostgreSQL database
-- ============================================================

-- MENTORS TABLE
CREATE TABLE IF NOT EXISTS mentors (
    mentor_id           SERIAL PRIMARY KEY,
    name                VARCHAR(100)        NOT NULL,
    email               VARCHAR(150)        UNIQUE NOT NULL,
    bio                 TEXT,                               -- used by NLP matching
    industry            VARCHAR(100),
    experience_years    INTEGER             DEFAULT 0,
    domains             TEXT[]              DEFAULT '{}',   -- ARRAY['AI','Leadership']
    skills              TEXT[]              DEFAULT '{}',   -- ARRAY['Python','SQL']
    mentorship_style    VARCHAR(50),                        -- 'structured'|'flexible'|'hands-on'|'advisory'
    availability_hours  INTEGER             DEFAULT 5,
    max_mentees         INTEGER             DEFAULT 3,
    location            VARCHAR(100),
    rating              DECIMAL(3,2)        DEFAULT 0.0,
    acceptance_rate     DECIMAL(5,4)        DEFAULT 0.5,
    total_mentees       INTEGER             DEFAULT 0,
    active              BOOLEAN             DEFAULT TRUE,
    created_at          TIMESTAMP           DEFAULT NOW(),
    updated_at          TIMESTAMP           DEFAULT NOW()
);

-- MENTEES TABLE
CREATE TABLE IF NOT EXISTS mentees (
    mentee_id           SERIAL PRIMARY KEY,
    name                VARCHAR(100)        NOT NULL,
    email               VARCHAR(150)        UNIQUE NOT NULL,
    goals               TEXT,                               -- free text, used by NLP matching
    industry            VARCHAR(100),
    current_level       VARCHAR(20)         DEFAULT 'beginner',
    desired_domains     TEXT[]              DEFAULT '{}',
    current_skills      TEXT[]              DEFAULT '{}',
    preferred_style     VARCHAR(50),
    availability_hours  INTEGER             DEFAULT 5,
    active              BOOLEAN             DEFAULT TRUE,
    created_at          TIMESTAMP           DEFAULT NOW(),
    updated_at          TIMESTAMP           DEFAULT NOW()
);

-- INTERACTIONS TABLE (used for model training)
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id      SERIAL PRIMARY KEY,
    mentor_id           INTEGER             REFERENCES mentors(mentor_id) ON DELETE CASCADE,
    mentee_id           INTEGER             REFERENCES mentees(mentee_id) ON DELETE CASCADE,
    status              VARCHAR(20)         DEFAULT 'pending',
    mentor_accepted     BOOLEAN,
    mentee_accepted     BOOLEAN,
    successful_match    BOOLEAN,
    overall_rating      DECIMAL(3,2),
    meetings_held       INTEGER             DEFAULT 0,
    created_at          TIMESTAMP           DEFAULT NOW(),
    completed_at        TIMESTAMP
);

-- INDEXES
CREATE INDEX IF NOT EXISTS idx_mentors_active       ON mentors(active);
CREATE INDEX IF NOT EXISTS idx_mentors_industry     ON mentors(industry);
CREATE INDEX IF NOT EXISTS idx_mentors_location     ON mentors(location);
CREATE INDEX IF NOT EXISTS idx_mentees_active       ON mentees(active);
CREATE INDEX IF NOT EXISTS idx_interactions_mentor  ON interactions(mentor_id);
CREATE INDEX IF NOT EXISTS idx_interactions_mentee  ON interactions(mentee_id);
CREATE INDEX IF NOT EXISTS idx_interactions_status  ON interactions(status);

-- ============================================================
-- MENTOR DATA
-- ============================================================

-- Deactivate any existing mentors before inserting fresh data
UPDATE mentors SET active = FALSE;

INSERT INTO mentors (
    name, email, skills, domains, experience_years, industry,
    mentorship_style, availability_hours, max_mentees, bio,
    location, rating, total_mentees, acceptance_rate, active
)
VALUES
(
    'Abba Musa Idris',
    'abba.idris@email.com',
    ARRAY['Public Speaking','Journalism','Storytelling','Leadership'],
    ARRAY['Public Speaking','Media','Youth Development'],
    8, 'Media', 'structured', 12, 3,
    'Experienced journalist, poet, and public speaking expert. Founder of Abdurabbihi Book Club and Noor Leadership Academy. Passionate about storytelling and mentoring young creatives.',
    'Kano', 4.6, 40, 0.85, TRUE
),
(
    'Sumayya Abdullahi Hussaini',
    'sumayya.hussaini@email.com',
    ARRAY['Advocacy','Policy Analysis','Public Accountability','Community Engagement'],
    ARRAY['Governance','Social Development','Public Policy'],
    6, 'Development', 'advisory', 10, 3,
    'Governance advocate focused on social justice and inclusive development. Works in policy, advocacy, and citizen engagement.',
    'Kaduna', 4.7, 35, 0.88, TRUE
),
(
    'Shamsuddeen Jibril',
    'shamsuddeen.jibril@email.com',
    ARRAY['Leadership','Robotics','Innovation','Entrepreneurship'],
    ARRAY['Engineering','Technology','Youth Mentorship'],
    7, 'Tech', 'hands-on', 15, 4,
    'Aerospace Engineering graduate and Co-founder of Vora Robotics Ltd. Founder of Mentminds mentoring young Africans in innovation and leadership.',
    'Kaduna', 4.8, 50, 0.90, TRUE
),
(
    'Dr. Hadiza Shettima Lawan',
    'hadiza.lawan@email.com',
    ARRAY['Sustainability','Climate Action','Agriculture','Research'],
    ARRAY['Environmental Science','Sustainable Development'],
    12, 'Academia', 'structured', 10, 3,
    'Dairy Scientist and Environmental Health Lecturer. Founder focused on sustainable agriculture and circular economy initiatives.',
    'Kano', 4.9, 60, 0.92, TRUE
),
(
    'Sajuda MIB',
    'sajuda.mib@email.com',
    ARRAY['Poetry','Creative Writing','Advocacy','Public Speaking'],
    ARRAY['Creative Arts','Social Advocacy'],
    5, 'Creative', 'flexible', 8, 3,
    'Poet and spoken word artist using storytelling to address social issues and inspire community change.',
    'Kaduna', 4.5, 25, 0.80, TRUE
),
(
    'Hafsat Asekegbe Mamudu',
    'hafsat.mamudu@email.com',
    ARRAY['Leadership','Civic Engagement','Project Coordination','Youth Mobilization'],
    ARRAY['Youth Development','Governance'],
    4, 'Nonprofit', 'structured', 8, 3,
    'Nursing student and civic engagement advocate. Founder of a leadership and personal development platform for students.',
    'Kaduna', 4.4, 20, 0.78, TRUE
),
(
    'Salmah Hassan Bizi',
    'salmah.bizi@email.com',
    ARRAY['Public Speaking','Youth Leadership','Girls Empowerment','Poetry'],
    ARRAY['Education','Creative Advocacy'],
    4, 'Education', 'hands-on', 9, 3,
    'Public speaker and spoken word poet dedicated to youth leadership and girls empowerment initiatives.',
    'Kaduna', 4.6, 22, 0.83, TRUE
),
(
    'Ameer Saeed',
    'ameer.saeed@email.com',
    ARRAY['Communication','Leadership','Peace Advocacy','Public Speaking'],
    ARRAY['Civic Leadership','Youth Engagement'],
    6, 'Civic', 'advisory', 12, 4,
    'Youth leader and public speaker with over 35 speaking engagements. Skilled in communication and civic mentorship.',
    'Kaduna', 4.7, 38, 0.87, TRUE
),
(
    'Abdulmalik Yahaya',
    'abdulmalik.yahaya@email.com',
    ARRAY['Creative Writing','Leadership','Peacebuilding','Advocacy'],
    ARRAY['Creative Leadership','Youth Development'],
    7, 'Creative', 'flexible', 10, 3,
    'Author, poet, and peace advocate promoting youth engagement and storytelling for social impact.',
    'Kano', 4.6, 30, 0.84, TRUE
)
ON CONFLICT (email) DO UPDATE SET
    name                = EXCLUDED.name,
    skills              = EXCLUDED.skills,
    domains             = EXCLUDED.domains,
    experience_years    = EXCLUDED.experience_years,
    industry            = EXCLUDED.industry,
    mentorship_style    = EXCLUDED.mentorship_style,
    availability_hours  = EXCLUDED.availability_hours,
    max_mentees         = EXCLUDED.max_mentees,
    bio                 = EXCLUDED.bio,
    location            = EXCLUDED.location,
    rating              = EXCLUDED.rating,
    total_mentees       = EXCLUDED.total_mentees,
    acceptance_rate     = EXCLUDED.acceptance_rate,
    active              = EXCLUDED.active,
    updated_at          = NOW();

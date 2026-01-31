import pandas as pd
from db import engine

query = """
SELECT
    i.successful_match,
    m.skills,
    m.domains,
    m.experience_years,
    m.mentorship_style,
    m.availability_hours AS mentor_hours,
    m.rating,
    m.acceptance_rate,
    t.goals,
    t.desired_domains,
    t.current_level,
    t.preferred_style,
    t.availability_hours AS mentee_hours
FROM interactions i
JOIN mentors m ON i.mentor_id = m.mentor_id
JOIN mentees t ON i.mentee_id = t.mentee_id
"""

df = pd.read_sql(query, engine)
print(df.head())

from sqlalchemy import create_engine, text

DB_URL = "postgresql://aihak:your_password_here@localhost:5432/mentor_ai"
engine = create_engine(DB_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT current_database();"))
    print(result.fetchone())
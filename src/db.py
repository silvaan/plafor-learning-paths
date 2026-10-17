"""Data Access Object for the PlaforEdu course/competency catalog."""
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "plaforedu.db"


class PlaforEduDB:
    def __init__(self, db_filepath: str | Path = DEFAULT_DB):
        self.db_filepath = str(db_filepath)
        self.conn = sqlite3.connect(self.db_filepath)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def fetch_one(self, query, params=()):
        self.cursor.execute(query, params)
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, query, params=()):
        self.cursor.execute(query, params)
        return [dict(row) for row in self.cursor.fetchall()]

    def get_course(self, course_id):
        return self.fetch_one(
            "SELECT id, name, description FROM courses WHERE id = ?", (course_id,)
        )

    def get_courses(self):
        return self.fetch_all("SELECT id, name, description FROM courses")

    def get_competency(self, competency_id):
        return self.fetch_one(
            "SELECT id, name, description FROM competencies WHERE id = ?",
            (competency_id,),
        )

    def get_competencies(self):
        return self.fetch_all("SELECT id, name, description FROM competencies")

    def get_courses_for_competency(self, competency_id):
        return self.fetch_all(
            """
            SELECT co.id, co.name, co.description
            FROM courses_competencies cc
            JOIN courses co ON cc.courseId = co.id
            WHERE cc.competencyId = ?
            """,
            (competency_id,),
        )

    def competencies_with_courses(self, min_courses: int = 1):
        """Return competencies that have at least `min_courses` candidate courses."""
        out = []
        for comp in self.get_competencies():
            courses = self.get_courses_for_competency(comp["id"])
            if len(courses) >= min_courses:
                comp = dict(comp)
                comp["n_candidates"] = len(courses)
                out.append(comp)
        return out

    def close(self):
        self.conn.close()

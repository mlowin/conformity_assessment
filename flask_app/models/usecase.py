import sqlalchemy as sa

class Usecase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String, unique=True, nullable=False)
    weights = db.Column(db.String)
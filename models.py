from app_test import db

class QuestionAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    answer_html = db.Column(db.Text)

    def __repr__(self):
        return f"QuestionAnswer(id={self.id}, question='{self.question}', answer='{self.answer}', answer_html='{self.answer_html}')"

from flask import Blueprint
from . import db, rds
from .methods import *
from .models import QuestionAnswer

modify = Blueprint("modify", __name__)


'''
curl -X POST -H "Content-Type: application/json" -d '{"que_id":que_id_from_fe}' "http://your-api-url/?satisfied=true"
API to take feedback from user and store their question along with model's generated answer
in a csv which later will be used to retrain the model for enhancing accuracy overtime.
'''
@modify.route('/feedback', methods=['POST'])
def feedback():
    query=request.get_json()
    satisfied = request.args.get('satisfied')
    que_id = int(query['que_id'])
    user_que = query['user_que']
    if satisfied == True:
        return jsonify({"response": "Data not saved"})   
    
    # Ensure that the user_q is not exactly similar to que at que_id in our database
    
    # actual_question_string = df_que_mapping.loc[que_id, 'question']
    
    actual_question = QuestionAnswer.query.get(int(que_id))

    if(actual_question != None and actual_question.question == user_que):
        return "No need to save since the exact same question already present!"

    ans_html = actual_question.answer_html  
    ans = actual_question.answer

    que = user_que
    ans = actual_question.answer
    ans_html = actual_question.answer_html
    id = actual_question.id
    new_entry = QuestionAnswer(question=que, answer=ans, answer_html=ans_html)

    # Add the new entry to the session
    db.session.add(new_entry)

    # Commit the changes to the database
    db.session.commit()    
    return "Success"

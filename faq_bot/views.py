from flask import Blueprint
from . import db, rds
from .methods import *

views = Blueprint("views", __name__)

@views.route("/", methods=['POST'])
def get_response():
    query=request.get_json()
    user_q=query["text"] 
    is_valid_que = check_if_question_valid(user_q)
    if(not is_valid_que):
        return jsonify({"response": "I'm sorry I don't understand your question. Please rephrase","intent": "not a question"})
    ans = rds.get(user_q)
    if ans is not None:
        # If the question is in the cache, return the answer
        answer=ans.decode()
        que_id = answer
        intent="redis"
    # If the question is not in the cache, search for similar questions in the cache
    else:
        similar_question = find_similar_question(user_q)
        if similar_question is not None:
            # If a similar question is found in the cache, retrieve the corresponding answer
            answer = rds.get(similar_question)
            answer=answer.decode()
            intent="redis_similar"                                    
        # Generate the answer
        else:
            texts = get_texts() # need to fix this call!
            results=similarity_search_batch(texts, user_q)   
            answer=get_content_after_string((results[0].page_content))
            rds.set(user_q, answer)
            intent="openai_api"
    return jsonify({"response":get_answer(answer),"intent":intent, "que_id": answer})
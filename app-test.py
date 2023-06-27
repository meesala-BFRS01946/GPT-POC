import os
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import NLTKTextSplitter
import pandas as pd 
import pickle
import redis
from fuzzywuzzy import fuzz        
import csv
from langchain.vectorstores import FAISS

rds = redis.Redis()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app=Flask(__name__)
mappings = pickle.load(open('Answer_mappings.pkl','rb'))
df=pd.DataFrame(mappings)
#nltk.download('punkt')   # Run this only once in the server
# Create a df to store the que_id and que_string mapping somewhere
df_que_mapping = pd.read_csv('Questions.csv')
# Set the 'question_id' column as the index
df_que_mapping.set_index('question_id', inplace=True)

with open('embedding_model.pkl', 'rb') as f:
    embedding = pickle.load(f)

@app.before_request
def initialize_variables():
    #  Initialising the trained_model once
    app.config['texts'] = get_texts()

def similarity_search_batch(texts_batch, user_question):
    db = FAISS.from_documents(texts_batch, embedding)
    return db.similarity_search(user_question)

def get_content_after_string(string):
    search_string = "question_id:"
    index = string.find(search_string)
    
    if index == -1:
        return None
    
    content = string[index + len(search_string):].strip()
    return content

    
def get_answer(Q_id):
    answer = df.loc[df['question_id'] == int(Q_id), 'Answer_html'].values[0]
    return answer

def get_texts():
    loader = CSVLoader(file_path='./Questions.csv',csv_args={'delimiter': ','})
    data = loader.load()
    text_splitter = NLTKTextSplitter(chunk_size=10000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)    
    return texts

'''
To handle cases when user says statements such as "Wrong answer", "I don't know"
TODO: Also need to keep track of the number of times the user has said a statement consequtively
'''
def check_if_question_valid(user_q):
    statements = [
        "That's incorrect.",
        "Wrong answer.",
        "I don't agree.",
        "Not what I was looking for.",
        "That's not what I meant.",
        "You didn't understand me.",
        "That's not the information I need.",
        "That's not the right response.",
        "I think you misunderstood.",
        "That's not helpful."
    ]

    for statement in statements:
        similarity_score = fuzz.partial_ratio(user_q, statement)
        if similarity_score >= 80:
            return False

    return True

def find_similar_question(question):
    # Retrieve all keys from the cache
    keys = rds.keys('*')
    # Iterate through the keys and find the most similar question
    highest_similarity = 70
    similar_question = None
    for key in keys:
        score=fuzz.ratio(question,key.decode())        
        if score > highest_similarity:
            highest_similarity = score
            similar_question = key.decode()
            print(highest_similarity)
    return similar_question


@app.route('/',methods=['POST'])
def train_doc():
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
        similar_question =find_similar_question(user_q)
        # similar_question =optimized_find_similar_question(user_q)
        if similar_question is not None:
            # If a similar question is found in the cache, retrieve the corresponding answer
            answer = rds.get(similar_question)
            answer=answer.decode()
            # ans = rds.get(user_q).decode()
            # que_id = ans
            intent="redis_similar"                                    
        # Generate the answer
        else:
            texts = app.config['texts']
            results=similarity_search_batch(texts, user_q)   
            answer=get_content_after_string((results[0].page_content))
            rds.set(user_q, answer)
            intent="openai_api"
    return jsonify({"response":get_answer(answer),"intent":intent, "que_id": answer})

'''
curl -X POST -H "Content-Type: application/json" -d '{"que_id":que_id_from_fe}' "http://your-api-url/?satisfied=true"
API to take feedback from user and store their question along with model's generated answer
in a csv which later will be used to retrain the model for enhancing accuracy overtime.
'''
@app.route('/feedback', methods=['POST'])
def feedback():
    query=request.get_json()
    satisfied = request.args.get('satisfied')
    que_id = int(query['que_id'])
    user_que = query['user_que']
    if satisfied == True:
        return jsonify({"response": "Data not saved"})   
    
    # Ensure that the user_q is not exactly similar to que at que_id in our database
    actual_question_string = df_que_mapping.loc[que_id, 'question']
    if(actual_question_string == user_que):
        return "No need to save since the exact same question already present!"

    # if que_id not in df["question_id"]:
    #     return "Que ID not present in Answer_mappings.pkl"

    ans_html = get_answer(que_id)    
    
    # Generate a new question_id and store the question_id, answer_html in a new csv
    max_question_id = df_que_mapping.index.max()
    new_que_id = max_question_id + 1

    # Define the file path
    file_path = 'feedback_data.csv'

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode and write the data
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the headers only if the file doesn't exist
        if not file_exists:
            writer.writerow(['question_id', 'user_question', 'answer_html'])
        
        # Write the new data
        writer.writerow([new_que_id, user_que, ans_html])
    return "Success"

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=int("5000"),debug=True)
from . import db, rds
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
from config import *
from flask_sqlalchemy import SQLAlchemy

mappings = pickle.load(open('Answer_mappings.pkl','rb'))
df=pd.DataFrame(mappings)

with open('embedding_model.pkl', 'rb') as f:
    embedding = pickle.load(f)

def similarity_search_batch(texts_batch, user_question):
    db_faiss = FAISS.from_documents(texts_batch, embedding)
    return db_faiss.similarity_search(user_question)

def get_content_after_string(string):
    search_string = "question_id:"
    index = string.find(search_string)  
    if index == -1:
        return None
    content = string[index + len(search_string):].strip()
    return content
    
def get_answer(q_id):
    answer = df.loc[df['question_id'] == int(q_id), 'Answer_html'].values[0]
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

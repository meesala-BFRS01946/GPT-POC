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

rds = redis.Redis()
db = SQLAlchemy()

def create_app():
    app=Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI    
    app.config['texts'] = get_texts()
    db.init_app(app)

    from .views import views
    from .modify import modify
    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(modify, url_prefix="/")

    return app



def get_texts():
    loader = CSVLoader(file_path='./Questions.csv',csv_args={'delimiter': ','})
    data = loader.load()
    text_splitter = NLTKTextSplitter(chunk_size=10000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)    
    return texts
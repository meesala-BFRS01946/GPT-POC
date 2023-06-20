import os
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain import ElasticVectorSearch
from werkzeug.middleware.profiler import ProfilerMiddleware
from langchain.text_splitter import NLTKTextSplitter
import pandas as pd 
import pickle
import re
import redis
from fuzzywuzzy import fuzz        
from elasticsearch import Elasticsearch
from langchain.vectorstores.elastic_vector_search import ElasticKnnSearch
from langchain.embeddings import ElasticsearchEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
import sys

import spacy
nlp=spacy.load('en_core_web_lg')
# nlp = pickle.load(open("nlp.pkl","rb"))
r = redis.Redis()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app=Flask(__name__)
mappings = pickle.load(open('Answer_mappings.pkl','rb'))
df=pd.DataFrame(mappings)
with open('embedding_model.pkl', 'rb') as f:
    embedding = pickle.load(f)
# class ExcludePostProfilerMiddleware:
#     def __init__(self, app, profile_dir):
#         self.app = app
#         self.profiler_middleware = ProfilerMiddleware(self.app, profile_dir=profile_dir)

#     def __call__(self, environ, start_response):
#         if environ["REQUEST_METHOD"] == "POST":
#             return self.app(environ, start_response)
#         return self.profiler_middleware(environ, start_response)
# app.wsgi_app = ExcludePostProfilerMiddleware(app.wsgi_app, profile_dir='./profile')

# opensearch_url = "https://vpc-stage-opensearch-zjknnaqog2fy5bp3vldq6zdmtq.ap-south-1.es.amazonaws.com:443"
# http_auth = ("stage-opensearch", "bnfh#475Hdphd#4583N")
opensearch_url = "https://localhost:9200",
http_auth = ("admin", "admin")
# docsearch = OpenSearchVectorSearch.from_documents(
#             texts[0:232],
#             embedding,
#             opensearch_url = "https://vpc-stage-opensearch-zjknnaqog2fy5bp3vldq6zdmtq.ap-south-1.es.amazonaws.com:443",
#             http_auth = ("stage-opensearch", "bnfh#475Hdphd#4583N"),
#             # opensearch_url = "https://localhost:9200",
#             # http_auth = ("admin", "admin"),
#             use_ssl = True,
#             verify_certs = False,
#             ssl_assert_hostname = False,
#             ssl_show_warn = False)
 #Function to perform similarity search for a batch of texts
def similarity_search_batch(texts_batch, user_question):
    docsearch = OpenSearchVectorSearch.from_documents(
    texts_batch,
    embedding,
    opensearch_url=opensearch_url,
    http_auth=http_auth,
    use_ssl = False,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    )
    return docsearch.similarity_search(user_question)


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

@app.route('/',methods=['POST'])
def train_doc():
    query=request.get_json()
    user_q=query["text"]
   # queryy = "give the answer to the user's query from the embeddings what you have got trained.The answers should be given from the Document provided to you  and if there is no answer from that document please return 'NULL' and nothing else should be returned apart from it.Please ensure to provide the answer in bullet points.The user query is {}".format(user_q)
    qq="Act as a FAQ answerer for a e-commerce company and please answer the question provided by the user and if there is no answer found to you as per your training please return 'NULL' and the user query is {} , please return 'NULL' if the question is not relevant to the document you have been trained".format(user_q)
    #r=qa.run(qq)
    ans = r.get(user_q)
    if ans is not None:
        # If the question is in the cache, return the answer
        answer=ans.decode()
        intent="redis"
    # If the question is not in the cache, search for similar questions in the cache
    else:
        similar_question =find_similar_question(user_q)
        if similar_question is not None:
            # If a similar question is found in the cache, retrieve the corresponding answer
            answer = r.get(similar_question).decode()   
            intent="redis_similar"                                    
            print("redis_answer")
            print(answer)
        # Generate the answer
        else:

            loader = CSVLoader(file_path='./Questions.csv',csv_args={'delimiter': ','})
            data = loader.load()
            text_splitter = NLTKTextSplitter(chunk_size=10000, chunk_overlap=0)
            texts = text_splitter.split_documents(data)
            inten="openai_api"
            #qa = RetrievalQA.from_chain_type(llm=OpenAI(batch_size=5), chain_type="map_reduce", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
            #sys.exit()

            ### This is overcome the http max request size exceeded error
            # batch_size = 200  # Adjust the batch size as needed
            # text_batches = []
            # text_batches.append(texts[0:232])
            # text_batches.append(texts[232:464])
            # # Perform similarity search for each batch
            # results = []
            # for batch in text_batches:
            #     batch_results = similarity_search_batch(batch,user_q)
            #     print("111111111111111111111111111111111111111111111111111111111111111111111111")
            #     results.extend(batch_results)
            results=similarity_search_batch(texts,user_q)           
            answer=get_content_after_string((results[0].page_content))

            print(answer)
            r.set(user_q, answer)
    return jsonify({"response":get_answer(answer),"intent":intent})
def find_similar_question(question):
    # Retrieve all keys from the cache
    keys = r.keys('*')
    # Iterate through the keys and find the most similar question
    highest_similarity = 70
    similar_question = None
    for key in keys:
        score=fuzz.partial_ratio(question,key.decode())        
        if score > highest_similarity:
            #highest_similarity = similarity
            similar_question = key.decode()
            break
            print(similarity)
            
            
    return similar_question


def optimized_find_similar_question(question):
    # Retrieve all keys from the cache
    keys = r.keys('*')
    # Sort the keys by similarity
    keys.sort(key=lambda key: fuzz.ratio(question, key.decode()))
    # Return the most similar question
    return keys[0].decode()

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=int("5000"),debug=True)
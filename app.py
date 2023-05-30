import os
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain import ElasticVectorSearch
from werkzeug.middleware.profiler import ProfilerMiddleware
import re

load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")

app=Flask(__name__)



class ExcludePostProfilerMiddleware:
    def __init__(self, app, profile_dir):
        self.app = app
        self.profiler_middleware = ProfilerMiddleware(self.app, profile_dir=profile_dir)

    def __call__(self, environ, start_response):
        if environ["REQUEST_METHOD"] == "POST":
            return self.app(environ, start_response)
        return self.profiler_middleware(environ, start_response)
app.wsgi_app = ExcludePostProfilerMiddleware(app.wsgi_app, profile_dir='./profile')

# def retrieve_value(idd):
#     query = {
#         "query": {
#             "ids": {
#                 "values": [idd]
#             }
#         }
#     }
#     result = es.search(index="my_index", body=query)
#     hits = result["hits"]["hits"]
#     if hits:
#         value = hits[0]["_source"]["value"]
#         return value
#     else:
#         return None

# def get_enclosed_values(sentence):
#     pattern = r"\{([^{}]+)\}"
#     matches = re.findall(pattern, sentence)
#     matches=[matches[i].strip() for i in range (len(matches))]
#     print(matches)
#     rep={}
#     for i in matches:
#         rep[str(i)]=retrieve_value(str(i))
#     print(rep)
#     for key, value in rep.items():
#         placeholder = "{" + key + "}"
#         sentence = sentence.replace(placeholder, value)

#     sentence = sentence.replace("{", "").replace("}", "")
#     return sentence

  
def get_content_after_string(string):
    search_string = "Answer_html:"
    index = string.find(search_string)
    
    if index == -1:
        return None
    
    content = string[index + len(search_string):].strip()
    return content

from elasticsearch import Elasticsearch
# Specify the Elasticsearch server URL
es = Elasticsearch(hosts=["http://localhost:9200"])



loader = CSVLoader(file_path='./que_ans (1).csv',csv_args={'delimiter': ','})
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
embedding = OpenAIEmbeddings()

#docsearch = Chroma.from_documents(texts, embeddings)
#qa = RetrievalQA.from_chain_type(llm=OpenAI(batch_size=5), chain_type="map_reduce", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
#rds = Redis.from_documents(texts, embeddings, redis_url="redis://localhost:6379",  index_name='link')
elastic_vector_search = ElasticVectorSearch(
elasticsearch_url="http://localhost:9200",
index_name="test_index",
embedding=embedding
        )
db = ElasticVectorSearch.from_documents(texts, embedding, elasticsearch_url="http://localhost:9200")
# qa = RetrievalQA.from_chain_type(llm=OpenAI(streaming=True,temperature=0), chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 1}))
#print(retrieve_value('img_id3'))
# @app.route('/',methods=['GET'])
# def home():
#     return "<center><h3>Welcome To Flask App For GPT</h3></center>"

@app.route('/enhance', methods=['POST'])
def prod_desc():
    data=request.get_json()
    prompt = "Please enhance the following product description {} based on the product name {} ".format(data['description'],data['product_name'])
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.314,
        max_tokens=256,
        top_p=0.54,
        frequency_penalty=0.44,
        presence_penalty=0.17)
    return jsonify({"response":str((response.choices[0].text).strip())})


@app.route('/',methods=['POST'])
def train_doc():
    query=request.get_json()
    user_q=query["text"]
   # queryy = "give the answer to the user's query from the embeddings what you have got trained.The answers should be given from the Document provided to you  and if there is no answer from that document please return 'NULL' and nothing else should be returned apart from it.Please ensure to provide the answer in bullet points.The user query is {}".format(user_q)
    qq="Act as a FAQ answerer for a e-commerce company and please answer the question provided by the user and if there is no answer found to you as per your training please return 'NULL' and the user query is {} , please return 'NULL' if the question is not relevant to the document you have been trained".format(user_q)
    #r=qa.run(qq)
    results = db.similarity_search(user_q)
    print("####################################################################")
    print(results)
    print("####################################################################")
   # er=retrieve_value("1234")
    return jsonify({"response":get_content_after_string((results[0].page_content)),"intent":"elk"})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=int("5000"),debug=True)
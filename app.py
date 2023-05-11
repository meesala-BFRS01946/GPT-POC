import os
import openai
import pickle
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd

load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return "<center><h3>Welcome To Flask App For GPT</h3></center>"

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


@app.route('/text',methods=['POST'])
def train_doc():
    query=request.get_json()
   # loader = TextLoader("FAQ")
  #  documents = loader.load()
    loader = CSVLoader(file_path='./faq.csv',csv_args={'delimiter': ','})
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
    user_q=query["question"]
    query = "give the answer to the user's query .The answers should be given from the Document provided to you and if there is no answer from that document please return 'sorry i don't know'.The user query is {}".format(user_q)
    return qa.run(query)

@app.route('/update', methods=['POST'])
def updateRow():
    body = request.get_json()
    file_path = 'faq.csv'
    _id = body['_id'] - 1
    column_name = body['column_name']
    value = body['value']
    df = pd.read_csv(file_path)
    df.loc[_id, column_name] = value
    df.to_csv(file_path, index = False)
    return "File Updated."

if __name__ == '__main__':
    app.run(debug=True)
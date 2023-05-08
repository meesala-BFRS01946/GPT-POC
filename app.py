import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return "welcome to Flask app for GPT"

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
    loader = TextLoader("FAQ")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
    user_q=query["question"]
    query = "give the answer to the user's query .Make sure if there are any sequential steps regarding the answer to the user's query,  then provide them in bulleted points .The user query is {}".format(user_q)
    return qa.run(query)

if __name__ == '__main__':
    app.run(debug=True)
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key =os.getenv("OPENAI_API_KEY")

app=Flask(__name__)

@app.route('/', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True)
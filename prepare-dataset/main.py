import configparser
import requests
import json
import csv

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config.get('API', 'key')
filename = 'que_ans.csv'

def main():
    # To get all categories
    category_url = 'https://shiprocket.freshdesk.com/api/v2/solutions/categories'
    categories = get_map(get_response(category_url), 'id', 'name')
    
    # Iterate over each category to get all folders
    for category_id in categories.keys():
        folder_url = 'https://shiprocket.freshdesk.com/api/v2/solutions/categories/' + category_id + '/folders'
        folders = get_map(get_response(folder_url), 'id', 'name')
        
        # Iterate over each folder to find the corresponding article
        i=1
        for folder_id in folders.keys():
            article_url = 'https://shiprocket.freshdesk.com/api/v2/solutions/folders/' + folder_id + '/articles'
            articles = get_articles(get_response(article_url))            
            total = len(articles)
            print(f"Iteration {i} of {total}")
            append_to_csv(category_id, folder_id, articles)
            i += 1        

def get_response(url):
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.get(url, headers=headers, auth=(api_key, 'X'))
    return response.content

def get_map(response, key, value):
    response = json.loads(response)
    result_dict = {str(item[key]): item[value] for item in response}
    return result_dict

def get_articles(response):
    response = json.loads(response)
    list = []
    for item in response:
        temp = (item['id'], item['title'], item['description_text'], item['description'])
        list.append(temp)
    return list

def append_to_csv(category_id, folder_id, result):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty
            writer.writerow(['C_id', 'F_id', 'Q_id', 'Question', 'Answer', 'Answer_html']) # Define the header
        for que_id, que, ans, ansHtml in result:
            if que[-1] != '?':
                writer.writerow([category_id, folder_id, que_id, que, ans, ansHtml])

main()
from elasticsearch import Elasticsearch
# Specify the Elasticsearch server URL
es = Elasticsearch(hosts=["http://localhost:9200"])

# Rest of the code remains the same
index_name = "my_index"
doc_type = "my_document"

def store_key_value(key, value,idd):
    document = {
        "key": key,
        "value": value
    }
    es.index(index=index_name ,id=idd, body=document)

def retrieve_value(key):
    query = {
        "query": {
            "match": {
                "key": key
            }
        }
    }
    result = es.search(index=index_name, body=query)
    hits = result["hits"]["hits"]
    if hits:
        value = hits[0]["_source"]["value"]
        return value
    else:
        return None
def retrieve_value(idd):
    query = {
        "query": {
            "ids": {
                "values": [idd]
            }
        }
    }
    result = es.search(index="my_index", body=query)
    hits = result["hits"]["hits"]
    print(hits)
    if hits:
        value = hits[0]["_source"]["value"]
        return value
    else:
        return None
def retrieve_value_1(idd):
    query = {
        "query": {
            "ids": {
                "values": [idd]
            }
        }
    }
    result = es.search(index="my_index", body=query)
    hits = result["hits"]["hits"]
    print(hits)
    if hits:
        value = hits[0]["_source"]["value"]
        return value
    else:
        return None
# Example usage
for i in range(1,11):
    store_key_value("img_id"+str(i),"www.imageurl"+str(i)+".com","img_id"+str(i))



retrieved_name = retrieve_value("1")
retrieved_age = retrieve_value('1234')

print("Retrieved name:", retrieved_name)
print("Retrieved age:", retrieved_age)

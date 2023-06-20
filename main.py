from elasticsearch import Elasticsearch
# Specify the Elasticsearch server URL
#es = Elasticsearch(hosts=["http://localhost:9200"])
es = Elasticsearch(
    cloud_id="my_deployment:dXMtZ292LWVhc3QtMS5hd3MuZWxhc3RpYy1jbG91ZC5jb20kODAwNDlkMmYxNTZmNDk0MWJlZjRiNTIyZTE3NDQ5NzkkMjc4MDdlMjllMThhNDc4ZmE2MmQ4MGM1NmI2NzQwN2I=",
    api_key="Y3I3SXBZZ0JWRTNEcHpWZU5GOFc6ck5nUnh6ZkVTMG1COU11aUpkZF9tdw==",
)

# Rest of the code remains the same
# index_name = "my_index"
# doc_type = "my_document"

# def store_key_value(key, value,idd):
#     document = {
#         "key": key,
#         "value": value
#     }
#     es.index(index=index_name ,id=idd, body=document)

# def retrieve_value(key):
#     query = {
#         "query": {
#             "match": {
#                 "key": key
#             }
#         }
#     }
#     result = es.search(index=index_name, body=query)
#     hits = result["hits"]["hits"]
#     if hits:
#         value = hits[0]["_source"]["value"]
#         return value
#     else:
#         return None
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
#     print(hits)
#     if hits:
#         value = hits[0]["_source"]["value"]
#         return value
#     else:
#         return None
# def retrieve_value_1(idd):
#     query = {
#         "query": {
#             "ids": {
#                 "values": [idd]
#             }
#         }
#     }
#     result = es.search(index="my_index", body=query)
#     hits = result["hits"]["hits"]
#     print(hits)
#     if hits:
#         value = hits[0]["_source"]["value"]
#         return value
#     else:
#         return None
# # Example usage
# for i in range(1,11):
#     store_key_value("img_id"+str(i),"www.imageurl"+str(i)+".com","img_id"+str(i))



# retrieved_name = retrieve_value("1")
# retrieved_age = retrieve_value('1234')

# print("Retrieved name:", retrieved_name)
# print("Retrieved age:", retrieved_age)




index_name = 'product_index'  # Choose a suitable index name

mapping = {
    'properties': {
        'product_id': {'type': 'integer'},
        'product_name': {'type': 'text'}
    }
}

es.indices.create(index=index_name, ignore=400)
es.indices.put_mapping(index=index_name, body=mapping)

def index_product(product_id, product_name):
    document = {
        'product_id': product_id,
        'product_name': product_name
    }
    es.index(index=index_name, body=document, id=product_id)


def search_products(query):
    body = {
        'query': {
            'match': {
                'product_name': query
            }
        }
    }
    response = es.search(index=index_name, body=body)
    hits = response['hits']['hits']
    return hits


results = search_products('A')
for hit in results:
    print(hit['_source']['product_name'])

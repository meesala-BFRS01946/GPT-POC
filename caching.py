import redis
from fuzzywuzzy import fuzz

r = redis.Redis()
r.set('what happens if i dont pay by invoice bill', 'it will lost')

# Get the value for a given key
value = r.get('what happens if i dont pay by invoice bill').decode()
print(value ) # Output: b'myvalue'


def find_similar_question(question):
    # Retrieve all keys from the cache
    keys = r.keys('*')
    # Iterate through the keys and find the most similar question
    highest_similarity = 0
    similar_question = None
    for key in keys:
        similarity = fuzz.ratio(question, key.decode())
        if similarity > highest_similarity:
            highest_similarity = similarity
            similar_question = key.decode()
    return similar_question

print(find_similar_question("what if i dont pay my bill"))
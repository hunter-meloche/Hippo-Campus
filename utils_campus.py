import os
import yaml
import shutil
from openai import OpenAI
import numpy as np
import time
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub


def get_key(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


cwd = os.getcwd()
client = OpenAI(api_key=get_key('api_key.txt'))
filename = filename = os.path.join(cwd, 'campus.yaml')
embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


class Memory:
    def __init__(self, message, summary):
        self.message = message
        self.summary = summary
        self.embedding = embedding_model([message]).numpy()


class Indice:
    def __init__(self, memory: Memory):
        self.summary = memory.message
        self.last_modified = float(time.time())
        self.embedding = memory.embedding.tolist()
        self.memories = [construct_memory_dict(memory, self.last_modified)]

    def to_dict(self):
        return {
            "summary": self.summary,
            "last_modified": self.last_modified,
            "embedding": self.embedding,
            "memories": self.memories
        }


# Creates a dict that can be inserted to the list of a topic's memories
def construct_memory_dict(memory: Memory, timestamp: float):
    memory_dict = {
        'memory': memory.message,
        'timestamp': timestamp,
        'embedding': memory.embedding.tolist()
    }
    return memory_dict


def open_yaml():
    # Initialize an empty list for index
    index = []
    if os.path.isfile(filename):
        # Load the data from the YAML file if it exists
        with open(filename, 'r') as file:
            loaded_data = yaml.safe_load(file)
            if loaded_data is not None:
                index = loaded_data    
    return index


def save_yaml(index: dict):
    with open(filename, 'w') as file:
        yaml.dump(index, file)


# Finds the most similar topic summary or topic memory through semantic similarity
def find_similar_embedding(query: Memory, choices: list):
    closest = -1
    highest_similarity = -1

    # Iterates through and compares similarity of topics or memories to query
    for i, item in enumerate(choices):
        similarity = cosine_similarity(query.embedding, np.array(item["embedding"]).reshape(1, -1))[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest = i

    return closest


# Finds memories related to query
def remember(query: str):
    memory_query = Memory(query, query) # Creates Memory object that holds query and query embedding
    results = search(memory_query)
    return results


# Searches for most releavnt topic and returns topic summary and most relevant topic memory
def search(memory_query: Memory):
    index = open_yaml()

    if not index:
        return "No memories exist in Campus yet."

    # Finds closest topic and topic memory
    closest_indice = find_similar_embedding(memory_query, index)
    closest_memory = find_similar_embedding(memory_query, list(index[closest_indice]["memories"]))

    results = f'''
    Most relevant summary found:
    {index[closest_indice]["summary"]}

    Most relevant memory found:
    {index[closest_indice]["memories"][closest_memory]["memory"]}
    '''

    return results


# Summarizes the messages between the user and assistant
def create_message_summary(message):
    system = '''
    Your job is to summarize interactions between a user and the assistant. 
    '''
    user = f'''
    Summarize the following interaction:
    
    {message}
    '''
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    response = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=prompt)
    return response.choices[0].message.content


# Creates a new topic for the new memory
def create_indice(memory, index):
    # Create new indice
    print(f"making new indice for {memory.message}")
    timestamp = float(time.time())
    new_indice = {
        "summary": memory.summary,
        "last_modified": timestamp,
        "embedding": memory.embedding.tolist(),
        "memories": [construct_memory_dict(memory, timestamp)]
    }
    # Add new indice
    index.append(new_indice)

    # Update the YAML file
    save_yaml(index)


# Absorbs new memory into found topic indice
def absorb(memory: Memory, index: list, closest_indice: int):
    system = '''
    You are a helpful assistant that creates salient summaries from existing summaries and new messages.
    Your response should be only the new summary itself and no other information.
    '''
    user = f'''
    Create a new summary from a given summary and a given message.
    Only answer with the new summary.
    Below are a few examples:

    Old Summary:
    The user mentioned that Star Trek: The Next Generation mainly takes place on the Starship Enterprise and that the chain of command in descending order is Captain Picard and Commander Riker.

    Message:
    The user mentioned that Lieutenant Commander Data is third in command aboard the Enterprise.
    The assistant asked if the user had a favorite episode.

    New Summary:
    The user mentioned that Star Trek: The Next Generation mainly takes place on the Starship Enterprise and that the chain of command in descending order is Captain Picard, Commander Riker and Lieutenant Commander Data.

    Old Summary:
    The user's guitar collection includes a Gibson Les Paul and a Fender Stratocaster and he has been playing for 16 years.
    The assistant asked if he had any other instruments.

    Message:
    The user mentioned that he just got a new saxaphone.
    The assistant asked if he would take lessons.

    New Summary:
    The user's instrument collection includes a Gibson Les Paul, Fender Stratocaster, and saxophone and he has been playing guitar for 16 years.
    The assistant asked if he had any other instruments and if he would take saxophone lessons.

    Old Summary:
    {index[closest_indice]["summary"]}

    Message:
    {memory.summary}

    New Summary:

    '''
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    # Create new summary that includes new message and create new topic embedding
    print(f"absorbing - {memory.message}")
    response = client.chat.completions.create(model='gpt-3.5-turbo-0125', messages=prompt)
    new_summary = response.choices[0].message.content.strip()
    new_summary_embedding = embedding_model([new_summary]).numpy().tolist()
    index[closest_indice]["summary"] = new_summary
    index[closest_indice]["last_modified"] = float(time.time())
    index[closest_indice]["summary_embedding"] = new_summary_embedding

    # Add new memory to list of topic's memories
    timestamp = float(time.time())
    memory_dict = construct_memory_dict(memory, timestamp)
    index[closest_indice]["memories"].append(memory_dict)

    # Update the YAML file
    save_yaml(index)


# Determines if the found topic is actually relevant to the new memory
def relevance(memory: Memory, index, closest_indice):
    system = '''
    You are a helpful assistant that determines if a summary is relevant to a given context.
    Your answer should only be "yes" or "no". You should not use any other words.
    '''
    user = f'''
    Determine if a given summary is relevant to a given context.
    Only answer "yes" or "no".
    Below are a few examples:

    Context:
    The user went on to explain that Bob Dylan was one of their favorite musicians and that they had seen him in concert multiple times.
    The user also mentioned that Blood on the Tracks was their favorite album.

    Summary:
    The user explained he has a bachelors in information technology because he didn't want to learn calculus.
    He is currently working as a senior DevOps engineer and dabbles in machine learning.
    The assistant asked what projects he is currently working on.

    Answer:
    no

    Context:
    The user explained that the Sony Playstation 5 (PS5) video game console was able to mitigate loading times by using high-end solid state drives (SSD).
    The user also explained that the video game he played most on the PS5 was Elden Ring and that he was excited for the upcoming DLC.

    Summary:
    The user mentioned that he enjoyed the Shadow of the Erdtree DLC for Elden Ring.
    The assistant asked what other games the user was looking forward to playing.

    Answer:
    yes

    Context:
    {index[closest_indice]["summary"]}

    Summary:
    {memory.summary}

    Answer:

    '''
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    # LLM prompt decides if memory fits indice summary and returns "yes" or "no" 
    response = client.chat.completions.create(model='gpt-3.5-turbo-0125', messages=prompt)
    result = response.choices[0].message.content.strip()
    print(f"relevance response - {result}")

    # Decide if memory should be absorbed into indice or if new indice should form
    if result == "yes":
        absorb(memory, index, closest_indice)
    else:
        create_indice(memory, index)


# Decides to organize the new memory into an existing topic or create a new one
def organize(memory: Memory):
    index = open_yaml()

    if not index:
        create_indice(memory, index)
        return

    # Finds most similar topic  
    closest_indice = find_similar_embedding(memory, index)
    # Decides if found topic is relevant enough to new memory
    relevance(memory, index, closest_indice)


# Takes in new memory and organizes it
def memorize(message: str):
    summary = create_message_summary(message)
    memory = Memory(message, summary) # Creates Memory object that holds message, summary and embedding
    organize(memory)

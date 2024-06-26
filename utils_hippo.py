import yaml
import requests
from urllib.parse import quote
import tiktoken
import time
import asyncio
import os
import aiohttp


TOKEN_LIMIT = 500


 # Get the current working directory and create the file path
cwd = os.getcwd()
filename = os.path.join(cwd, 'hippo.yaml')


def open_yaml():
    # Initialize an empty list for data
    data = []
    if os.path.isfile(filename):
        # Load the data from the YAML file if it exists
        with open(filename, 'r') as file:
            loaded_data = yaml.safe_load(file)
            if loaded_data is not None:
                data = loaded_data    
    return data


def save_yaml(data):
    with open(filename, 'w') as file:
        yaml.dump(data, file)


def memorize(message: str):
    data = open_yaml()

    # Create a new entry
    new_entry = {
        'message': message,
        'timestamp': float(time.time())
    }

    # Append the new entry to the list
    data.append(new_entry)

    # Write the updated data back to the YAML file
    save_yaml(data)


async def send_to_campus(entry: dict):
    formatted_memory = quote(entry['message'])
    response = requests.post(f"http://localhost:8001/memorize?message={formatted_memory}")
    print(f"Memory sent to Campus: {entry['message']}\n")


def pop_oldest(data, reversed_i: int):
    print(f"Pop oldest: {reversed_i}")
    loop = asyncio.get_event_loop()
    # Check if data is not empty before proceeding
    if not data:
        print("No data to pop.")
        return
    # Takes oldest entries out of data
    if reversed_i < 0:
        old_memory = data.pop(0)
        loop.create_task(send_to_campus(old_memory))
    else:
        for i in range(0, reversed_i):
            if not data:  # Check again as data might become empty during the loop
                print("No more data to pop.")
                break
            if i == reversed_i:
                break
            old_memory = data.pop(0)
            loop.create_task(send_to_campus(old_memory))

    save_yaml(data)


def remember():
    data = open_yaml()

    if not data:
        return "Hippo does not have any memories yet."

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    memories = handle_corpus(data, encoding)

    return memories


def handle_corpus(data, encoding):
    corpus_tokens = 0
    # gets the length of data as a string
    data_length = len(str(data))
    print(f"Data length: {data_length}")
    corpus = []

    for i, memory in enumerate(reversed(data)):
        corpus_tokens += len(encoding.encode(memory["message"]))

        if (corpus_tokens + 4) > TOKEN_LIMIT:
            reversed_i = data_length - i - 1
            print(f"reversed_i: {reversed_i}")
            pop_oldest(data, reversed_i)
            break

        corpus.insert(0, memory["message"])

    print(f"Corpus tokens: {corpus_tokens}")
    separator = '\n\n'
    memories = separator.join(corpus)

    return memories
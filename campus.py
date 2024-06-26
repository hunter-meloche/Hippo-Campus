import openai
import utils_campus as utils
from fastapi import FastAPI

app = FastAPI()

@app.post("/memorize")
async def add_message(message: str):
    print('\n\nMEMORIZING -', message)
    utils.memorize(message)
    return {"detail": "Memorized!"}

@app.get("/remember")
async def search(query: str):
    print('\n\nSEARCHING CAMPUS')
    result = utils.remember(query)
    return {"results": result}
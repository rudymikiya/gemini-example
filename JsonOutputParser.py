import os

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# read the api key from MyApiKey.txt
with open('MyApiKey.txt', 'r') as file:
    api_key = file.read().strip()
    print(f"API key: {api_key}")
    os.environ['GOOGLE_API_KEY'] = api_key


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


if __name__ == "__main__":
    joke_query = "Tell me a joke."
    parser = PydanticOutputParser(pydantic_object=Joke)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = (prompt | ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest") | parser)
    res = chain.invoke({"query": joke_query})
    print(res.setup)
    print(res.punchline)

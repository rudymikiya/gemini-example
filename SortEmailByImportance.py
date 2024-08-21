import os

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# read the api key from MyApiKey.txt
with open('MyApiKey.txt', 'r') as file:
    api_key = file.read().strip()
    print(f"API key: {api_key}")
os.environ["GOOGLE_API_KEY"] = api_key

# write the email summaries to a json file
def write_summary(summary: str, output_file: str):
    with open(output_file, "w") as file:
        file.write(summary)

if __name__ == "__main__":
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest")
    with open("out/email_summaries.json", 'r') as file:
        # Load the JSON data from the file
        json_string = file.read()
    template = """Below is a list of email summaries and format is a json. Please order the summaries in descending order of importance. The result should have summary and fileName:
            {context}

            Treat it as important: {important_criteria}
            You should response like a secretary and the content is not a json. The start should be like 'These are important the unread emails recently:' and use arabic numerals for list. Put the content of fileName and summary in different lines.
            """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
            {"context": lambda x: json_string,
             "important_criteria": RunnablePassthrough()}
            | prompt
            | model
    )

    res = retrieval_chain.invoke("from <selfimprovement-space@quora.com> is important.");
    print(res.content)
    write_summary(res.content, "./out/summary.md")

import json
import os

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# read the api key from MyApiKey.txt
with open('MyApiKey.txt', 'r') as file:
    api_key = file.read().strip()
    print(f"API key: {api_key}")
os.environ["GOOGLE_API_KEY"] = api_key


class EmailSummary(BaseModel):
    # fromEmail: str = Field(description="The email address of the sender")
    # toEmails: str = Field(description="The email addresses of the recipients")
    # fileName: str = Field(description="The name of the email file")
    summary: str = Field(description="The summary of the email")


# find the from and to email addresses from the email
def find_recipient_by_line_start(email: str, line_start: str) -> str:
    for line in email.splitlines():
        if line.startswith(line_start):
            return line.split(":")[1].strip()


# write the email summaries to a json file
def write_email_summaries(_email_summaries: list, output_file: str):
    with open(output_file, "w") as _file:
        _file.write(json.dumps(_email_summaries))


if __name__ == "__main__":
    email_root_dir = "emails"
    email_summaries = []
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest")
    for root, dirs, files in os.walk(email_root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            loader = TextLoader(file_path)
            data = loader.load()
            fromEmail = find_recipient_by_line_start(data[0].page_content, "From:")
            toEmail = find_recipient_by_line_start(data[0].page_content, "To:")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            all_splits = text_splitter.split_documents(data)
            vectorstore = FAISS.from_documents(documents=all_splits,
                                               embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
            # https://www.mirascope.io/post/langchain-prompt-template
            parser = PydanticOutputParser(pydantic_object=EmailSummary)
            template = """Summarize this email based only on the following context:
            {context}

            The result format: {resultFormat}
            """
            prompt = ChatPromptTemplate.from_template(template)
            retrieval_chain = (
                    {"context": vectorstore.as_retriever(),
                     "resultFormat": RunnablePassthrough()}
                    | prompt
                    | model
                    | parser
            )

            res = retrieval_chain.invoke("Generate a json object as result."
                                         " Summarize this email and put it into 'summary' field. ")
            print(res)
            email_summaries.append(
                {"fromEmail": fromEmail, "toEmails": toEmail, "fileName": file, "summary": res.summary})

    write_email_summaries(email_summaries, "./out/email_summaries.json")

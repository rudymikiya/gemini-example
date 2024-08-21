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
    # toEmails: set[str] = Field(description="The email addresses of the recipients")
    # fileName: str = Field(description="The name of the email file")
    summary: str = Field(description="The summary of the email")


if __name__ == "__main__":
    email_root_dir = "emails"
    sample_files = []
    email_content = []
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest")
    for root, dirs, files in os.walk(email_root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            loader = TextLoader(file_path)
            data = loader.load()
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
                    {"context": vectorstore.as_retriever(), "fileName": lambda x: file, "resultFormat": RunnablePassthrough()}
                    | prompt
                    | model
                    | parser
            )

            data = retrieval_chain.invoke("Generate a json object as result."
                                          " Summarize this email and put it into 'summary' field. ")
            print(data)

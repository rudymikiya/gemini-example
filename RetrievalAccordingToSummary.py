import os

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

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
def write_retrieval(retrieval: str, output_file: str):
    with open(output_file, "w") as file:
        file.write(retrieval)


question = "I want the know the information about 'HOSTDARE'"

if __name__ == "__main__":
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest")
    with open("out/summary.md", 'r') as file:
        # Load the JSON data from the file
        summary = file.read()
    template = """Below are some email summaries, each item is a file name + its summary. Please return the fileName according to the Request:
            {context}

            Request: {request}
            Only return the fileName is enough.
            """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
            {"context": lambda x: summary,
             "request": RunnablePassthrough()}
            | prompt
            | model
    )

    res = retrieval_chain.invoke(question);
    most_likely_file_name = res.content
    print(res.content)

    with open("out/email_summaries.json", 'r') as file:
        # Load the JSON data from the file
        json_string = file.read()
    template = """Below is a list of email summaries and format is a json. Please find the most likely fileName according to the fileName I pass:
            {context}

            fileName: {fileName}
            Only return the plain fileName is enough.
            """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
            {"context": lambda x: json_string,
             "fileName": RunnablePassthrough()}
            | prompt
            | model
    )

    res = retrieval_chain.invoke(most_likely_file_name);

    print(res.content)

    email_path = "./emails/" + res.content.strip()

    loader = TextLoader(email_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(documents=all_splits,
                                       embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    template = """Below is a email, please answer the question according to the email:
            {context}

            Question: {question}
            Give more detailed answer.
            """
    prompt = ChatPromptTemplate.from_template(template)
    retrieval_chain = (
            {"context": vectorstore.as_retriever(),
             "question": RunnablePassthrough()}
            | prompt
            | model
    )

    res = retrieval_chain.invoke(question)
    print(res.content)

    write_retrieval(res.content, "./out/retrieval.txt")

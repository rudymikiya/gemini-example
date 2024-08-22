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
    with open(output_file, "w") as _file:
        _file.write(retrieval)


question = "I want the know the details about Production Issue"


def get_file_name_from_summary(_model: ChatGoogleGenerativeAI) -> str:
    with open("out/summary.md", 'r') as _file:
        # Load the JSON data from the file
        _summary = _file.read()
    _template = """Below are some email summaries, each item is a file name + its summary. Please return the fileName according to the Request:
            {context}

            Request: {request}
            Only return the fileName is enough.
            """
    _prompt = ChatPromptTemplate.from_template(_template)

    _retrieval_chain = (
            {"context": lambda x: _summary,
             "request": RunnablePassthrough()}
            | _prompt
            | _model
    )

    _res = _retrieval_chain.invoke(question)
    _most_likely_file_name = _res.content
    print("Most likely file name:", _most_likely_file_name)
    return _most_likely_file_name


def get_exact_filename(_model: ChatGoogleGenerativeAI, _most_likely_file_name: str) -> str:
    with open("out/email_summaries.json", 'r') as _file:
        # Load the JSON data from the file
        _json_string = _file.read()
    _template = """Below is a list of email summaries and format is a json. Please find the most likely fileName according to the fileName I pass:
            {context}

            fileName: {fileName}
            Only return the plain fileName is enough.
            """
    _prompt = ChatPromptTemplate.from_template(_template)

    _retrieval_chain = (
            {"context": lambda x: _json_string,
             "fileName": RunnablePassthrough()}
            | _prompt
            | _model
    )

    _res = _retrieval_chain.invoke(_most_likely_file_name)
    _file_name = _res.content.strip()
    print("File name:", _file_name)
    return _file_name


if __name__ == "__main__":
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest")

    most_likely_file_name = get_file_name_from_summary(model)
    file_name = get_exact_filename(model, most_likely_file_name)

    email_path = "./emails/" + file_name
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

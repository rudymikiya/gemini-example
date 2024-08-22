import os

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


class EmailRetrieval(object):

    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        self.model = ChatGoogleGenerativeAI(
            model=model_name)

    def retrieve(self, request: str, summary_output_file: str, processed_output_file: str, email_folder: str) -> str:
        most_likely_file_name = self.__get_file_name_from_summary(summary_output_file, request)
        file_name = self.__get_exact_filename(processed_output_file, most_likely_file_name)

        email_path = os.path.join(email_folder, file_name)
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
                | self.model
        )

        res = retrieval_chain.invoke(request)
        return res.content

    def __get_file_name_from_summary(self, summary_output_file: str, request: str) -> str:
        with open(summary_output_file, 'r') as file:
            # Load the JSON data from the file
            summary = file.read()
        template = """Below are some email summaries, each item is a file name + its summary. Please return the fileName according to the Request:
                {context}
    
                Request: {request}
                Must return one and only return the fileName is enough.
                """
        prompt = ChatPromptTemplate.from_template(template)

        retrieval_chain = (
                {"context": lambda x: summary,
                 "request": RunnablePassthrough()}
                | prompt
                | self.model
        )

        res = retrieval_chain.invoke(request)
        most_likely_file_name = res.content
        print("Most likely file name:", most_likely_file_name)
        return most_likely_file_name

    def __get_exact_filename(self, processed_output_file: str, most_likely_file_name: str) -> str:
        with open(processed_output_file, 'r') as file:
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
                | self.model
        )

        res = retrieval_chain.invoke(most_likely_file_name)
        file_name = res.content.strip()
        print("File name:", file_name)
        return file_name

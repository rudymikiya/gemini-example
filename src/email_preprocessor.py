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


class EmailSummary(BaseModel):
    # fromEmail: str = Field(description="The email address of the sender")
    # toEmails: str = Field(description="The email addresses of the recipients")
    # fileName: str = Field(description="The name of the email file")
    summary: str = Field(description="The summary of the email")


class EmailPreprocessor(object):

    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        self.model = ChatGoogleGenerativeAI(
            model=model_name)

    def generate_summaries(self, email_folder: str, output_file: str):
        email_summaries = []
        for root, dirs, files in os.walk(email_folder):
            for file in files:
                file_path = os.path.join(root, file)
                loader = TextLoader(file_path)
                data = loader.load()
                from_email = self.__find_recipient_by_line_start(data[0].page_content, "From:")
                to_email = self.__find_recipient_by_line_start(data[0].page_content, "To:")
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
                        | self.model
                        | parser
                )
                # About how retrival works
                # step1: The goal of this function is to generate a dictionary with two keys: context and question, used to pass the context to the prompt template.
                #        The 'context' is generated by retriever with the message passed by 'important_criteria context'. Retriever will get from the trucks and embedings and calulate the final context and set it as 'context' property.
                # step2: Pass the dictionary generated in the first step to the prompt template for prompt template formatting
                # step3: Pass the formatted prompt of the prompt template to the model (model)
                # step4: Pass the model call results to the output parser StrOutputParser
                res = retrieval_chain.invoke("Generate a json object as result."
                                             " Summarize this email and put it into 'summary' field. ")
                print(res)
                email_summaries.append(
                    {"fromEmail": from_email, "toEmails": to_email, "fileName": file, "summary": res.summary})

        self.__write_email_summaries(email_summaries, output_file)

        # find the from and to email addresses from the email

    def __find_recipient_by_line_start(self, email: str, line_start: str) -> str:
        for line in email.splitlines():
            if line.startswith(line_start):
                return line.split(":")[1].strip()

    # write the email summaries to a json file
    def __write_email_summaries(self, email_summaries: list, output_file: str):
        with open(output_file, "w") as _file:
            _file.write(json.dumps(email_summaries))

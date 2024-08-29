
from langchain_community.llms import Ollama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
#from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

from pathlib import Path


class ChatPDF:
    context = None
    chain = None
    old_temperature = 0.01
    
    def __init__(self):
        self.model = Ollama(model="qwen2:7b", temperature=self.old_temperature, num_ctx=50000)
        self.prompt = PromptTemplate.from_template("""
            You are an assistant to retrieve the data from documents.
            Please answer the following question, using the following documents. 
            Please do not answer if the answer is not in the documents. Do not make up an answer.
            Documents:

            {context}

            Question:

            {question}

            Write your answer in Russian language:
            Make sure your answer is just the answer, with no commentary.
            Start!
            
            """ 
            #Ты ассистент для ответов на вопросы по заданному контексту. 
            # Ты знаешь только то, что представлено в контексте. Если ты не нашел ответа в контексте, скажи, что ты не знаешь ответа. Отвечай максимально коротко одним предложением максимум.
            # Контекст: {context}
            # Вопрос: {question}
            # Ответ: """
        )

    def change_temperature(self, temperature):
        if self.old_temperature != temperature:
            print(f"Меняем температуру с {self.old_temperature} на {temperature}\n")
            self.model = Ollama(model="qwen2:7b", temperature=temperature, num_ctx=50000)
            self.old_temperature = temperature

    def ingest(self, doc_file_path: str):       
        def format_docs(docs):  
            retrieved_docs_text = [doc.page_content for doc in docs]  
            context = "".join(retrieved_docs_text)
            return context

        suffix = Path(doc_file_path).suffix
        if suffix == ".docx":
            loader = Docx2txtLoader(doc_file_path) 
        else:
            loader = PyPDFLoader(doc_file_path) 
        docs = loader.load()
        self.context = format_docs(docs)
        
        self.chain = ({"context": lambda x: self.context, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())      

    def ask(self, query: str):
        if not self.chain:
            return "Сначала добавьте документ."
        return self.chain.invoke(query)
    
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
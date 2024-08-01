
from langchain_community.llms import Ollama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from pathlib import Path


class ChatPDF:
    context = None
    chain = None
    
  
    def __init__(self):
        self.model = Ollama(model="qwen2:7b", temperature=0.001, num_ctx=50000)
        self.prompt = PromptTemplate.from_template(""" Ты ассистент для ответов на вопросы по заданному контексту. 
            Если ты не нашел ответа в контексте, скажи, что ты не знаешь ответа. Отвечай коротко.
            Контекст: {context}
            Вопрос: {question}
            Ответ: """
        )

    def ingest(self, doc_file_path: str):       
        def format_docs(docs):  
            retrieved_docs_text = [doc.page_content for doc in docs]  
            context = "\nВыбранный контекст:\n"
            context += "".join([f"Часть контекста {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
            return context

        suffix = Path(doc_file_path).suffix
        print(suffix)
        if suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(doc_file_path) 
        else:
            loader = UnstructuredPDFLoader(doc_file_path) 
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
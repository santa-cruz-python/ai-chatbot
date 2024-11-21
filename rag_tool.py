from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_ollama import ChatOllama
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_core.tools import tool
import json


class RAGTool():
    vectorstore = None

    def __init__(self):
        # split the text in the page into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        # embeddings = OllamaEmbeddings(model="llama3.2")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # self.model = ChatOllama(
        #     model="llama3.2",
        #     temperature=0.5
        # )
        self.model = OpenAI(
            temperature=0.5,
            model="gpt-4o-mini",
            max_tokens=200
        )

        self.prompt = ChatPromptTemplate.from_template(
            """Considering the following context: 
            {docs}
            answer the question"""
        )

    def load(self, url):
        # load data from webpage
        loader = WebBaseLoader(url)
        # load the data
        data = loader.load()
        # print(data)
        all_splits = self.text_splitter.split_documents(data)
        # all_splits = self.html_splitter.split_text(data)
        print(json.dumps([d.page_content for d in all_splits], indent=2))
        # create the document store
        # Note that with from document, a new index is created each time. To add documents use .add_documents
        self.vectorstore = Chroma.from_documents(collection_name="docs", documents=all_splits, embedding=self.embeddings)


    def predict(self, question):

        def format_docs(docs):
            """Extract content from document retrieved"""
            return "\n\n".join(doc.page_content for doc in docs)

        chain = {"docs": format_docs} | self.prompt

        retriever = self.vectorstore.as_retriever(k=10)
        _filter = LLMChainFilter.from_llm(self.model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_filter, base_retriever=retriever
        )

        # docs = retriever.invoke(question[-1])
        docs = compression_retriever.invoke(question[-1])
        print("RETRIEVED DOCS")
        print(json.dumps([d.page_content for d in docs], indent=2))
        response = chain.invoke(docs)

        return response

url_cache = {}
rag = RAGTool()

@tool
def rag_tool(question: str, url: str):
    """Use this tool to ask questions about the content of a given URL

    Args:
    question: The question from the user.
    url: the URL to parse and search for answers.
    """

    print(question)
    print(url)
    if url not in url_cache.keys():
        rag.load(url)
        url_cache[url] = True

    return rag.predict(question)



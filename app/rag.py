from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from app.config import get_urls_for_topic

load_dotenv()  # Load environment variables

class RAGPipeline:
    def __init__(self, api_key: str = None):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # Updated based on your available models
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash-latest",  # Updated based on your available models
            temperature=0.3,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.persist_dir = "chroma_db"

    def build_vectorstore(self, topic, subtopic):
        urls = get_urls_for_topic(topic, subtopic)
        if not urls:
            if topic == "frontend" and subtopic == "react":
                urls = [
                    "https://react.dev/learn",
                    "https://react.dev/reference/react"
                ]
            else:
                urls = ["https://developer.mozilla.org/en-US/docs/Web/JavaScript"]
        
        try:
            loader = WebBaseLoader(urls)
            docs = loader.load()
            if not docs:
                raise ValueError("No documents loaded")
            
            vectordb = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_dir)
            return vectordb.as_retriever()
        except Exception as e:
            raise ValueError(f"Error building vectorstore: {str(e)}")

    def adaptive_retrieve(self, retriever, query, threshold=0.75):
        docs = retriever.get_relevant_documents(query)
        if not docs or len(docs) < 2:
            docs = retriever.get_relevant_documents("more general " + query)
        return docs

    def generate_mcq(self, query):
        retriever = self.build_vectorstore("temp", "temp")  # dummy to initialize
        docs = self.adaptive_retrieve(retriever, query)
        context = "\n".join([doc.page_content for doc in docs[:2]])
        prompt = f"""
        Based on the context below, generate a multiple-choice question with 4 options and the correct answer.

        Context:
        {context}

        Format:
        Question: <Question>
        A. <Option A>
        B. <Option B>
        C. <Option C>
        D. <Option D>
        Correct: <Correct Option>
        """
        return self.llm.invoke(prompt).content.strip()

    def auto_mode(self, topic, subtopic):
        retriever = self.build_vectorstore(topic, subtopic)
        queries = [f"Beginner {subtopic}", f"Intermediate {subtopic}", f"Advanced {subtopic}"]
        for i, query in enumerate(queries, start=1):
            mcqs = [self.generate_mcq(query) for _ in range(2)]
            if all("Question:" in m for m in mcqs):
                return i, mcqs
        return 1, [self.generate_mcq(f"{subtopic} basic")]

    def test_api_connection(self):
        """Test if the API connections are working"""
        try:
            # Test embeddings API
            test_text = "This is a test sentence."
            embedding_result = self.embeddings.embed_query(test_text)
            if not embedding_result:
                return "Embedding API test failed"

            # Test LLM API
            test_prompt = "Write 'Hello, API is working!' if you can read this."
            llm_result = self.llm.invoke(test_prompt)
            if not llm_result:
                return "LLM API test failed"

            return "All API connections are working"
            
        except Exception as e:
            return f"API test failed: {str(e)}"

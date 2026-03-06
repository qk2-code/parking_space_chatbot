import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def setup_rag_system(file_path: str):
    loader = TextLoader(f"data/{file_path}", encoding='utf-8')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)

    client = chromadb.EphemeralClient()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client=client,
        collection_name="parking_collection"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """you are a parking space helper. 
    use the given context to answer the question. 
    if the answer is not in the context, say "I don't know".

    context: {context}
    question: {question}

    answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 6. Побудова RAG Chain (LCEL)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}

            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    chain = setup_rag_system("parking.txt")
    response = chain.invoke("Які години роботи парковки та де знаходиться в'їзд?")
    print(f"Bot: {response}")

import chromadb
import os
from alive_progress import alive_bar
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI


# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def main():
    # Load environment variables and fetch openai api key
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")

    # Setup Embedding Function instance
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key, model_name="text-embedding-3-small"
    )

    # Initialize the Chroma client with persistent storage
    chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
    collection_name = "document_qa_collection"
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=openai_ef
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_key)

    # Function to generate embeddings using OpenAI API
    def get_openai_embedding(text):
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = response.data[0].embedding
        return embedding

    # Load documents from the directory
    direcotry = "./news_articles/"
    documents = load_documents_from_directory(direcotry)
    print(f"Loaded {len(documents)} from {direcotry}")

    # Split documents into chunks
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})
    print(f"Splited into {len(chunked_documents)} chunks")

    # Generate embeddings for the document chunks
    with alive_bar(
        len(chunked_documents), title="==== Generating embeddings ===="
    ) as bar:
        for i, doc in enumerate(chunked_documents):
            doc["embeddings"] = get_openai_embedding(doc)
            bar()

    # Upsert doccuments and embeddings into Chroma
    with alive_bar(
        len(chunked_documents), title="==== Adding docs to Chroma ===="
    ) as bar:
        for i, doc in enumerate(chunked_documents):
            collection.upsert(
                ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embeddings"]]
            )
            bar()

    def query_documents(question, n_results=2):
        results = collection.query(query_texts=question, n_results=n_results)
        relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
        print("==== Returning relevant chunks ====")
        return relevant_chunks

    def generate_response(question, relevant_chunks):
        context = "\n\n".join(relevant_chunks)
        prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of "
            "retrieved context to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        answer = response.choices[0].message
        return answer

    question = "databricks　とは？"
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(answer)


if __name__ == "__main__":
    main()

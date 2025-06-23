from sentence_transformers import SentenceTransformer
import chromadb
import hashlib

class EmbeddingHelper:

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(name="pdf_embeddings")
        except:
            self.collection = self.chroma_client.create_collection(name="pdf_embeddings")

    def get_embedding(self,texts):
        """
        It helps to get embedding from texts.
        :param texts:
        :return: embedding of texts
        """
        embedding = self.embedder.encode(texts, convert_to_tensor=True)
        return embedding

    def store_embedding(self, chunks, embedding):
        """
        It stores embedding from chunks.
        :param chunks: list of chunks
        :param embedding: embedding of chunks
        """
        def get_hash(chunk):
            return hashlib.md5(chunk.encode()).hexdigest()

        metadata = [{"text": chunk} for chunk in chunks]
        ids = [get_hash(chunk) for chunk in chunks]
        self.collection.upsert(embeddings=embedding.tolist(), documents=chunks, metadatas=metadata, ids=ids)

    def get_top_n_results(self, query, n):
        """
        It returns top n results from query.
        :param query: text to search for
        :param n: int, number of results to return
        :return: list, list of top n results
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        results = self.collection.query(query_embedding.tolist(), n_results=n)
        return results["documents"][0]
from flask import Flask, render_template, request, jsonify
from Helpers.pdfHelper import PdfHelper
from Helpers.embeddingHelper import EmbeddingHelper
from Helpers.modelHelper import RAGModel

app = Flask(__name__)
embedding_helper = EmbeddingHelper()
model = RAGModel()

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    # Get the user input/question from the form
    query = request.form["msg"]
    # Get the top n(which is selected as 5) results from the embedding helper
    docs = embedding_helper.get_top_n_results(query, 5)
    # Generate the answer using the RAG model
    answer = model.generate_answers(" ".join(docs), query)
    # Return the answer as a JSON response to display in the chat window
    return jsonify({"response": answer})


if __name__ == '__main__':
    # Load and process the PDF files
    # This should be done once and the results stored as .txt for later use
    pdf_paths = PdfHelper.get_pdf_names_from_folder('dataset')
    texts = PdfHelper.extract_text_from_pdfs(pdf_paths, save_output=True)
    # Split the texts into chunks for embedding
    chunks, metadata = PdfHelper.chunk_texts(texts)
    # Store the chunks and metadata for later use
    embedding = embedding_helper.get_embedding(chunks)
    # Store the embeddings in a database or file for later retrieval
    embedding_helper.store_embedding(chunks, embedding)

    app.run(debug=True)
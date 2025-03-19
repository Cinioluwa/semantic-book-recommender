import pandas as pd
import numpy as np
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, jsonify

# ---------------------------
# Load and Prepare the Dataset
# ---------------------------
df = pd.read_csv('prod_dataset_combined_embeddings.csv')
df['Description'] = df['Description'].fillna('')

def convert_embedding(embedding_str):
    """
    Convert a string representation of an embedding into a NumPy array.
    Removes any "np.float32(...)" wrappers and "array(...)" wrappers before converting.
    """
    # Remove np.float32 wrappers using regex.
    if "np.float32(" in embedding_str:
        embedding_str = re.sub(r'np\.float32\(([^)]+)\)', r'\1', embedding_str)
    # Remove "array(" wrapper if present.
    if embedding_str.startswith("array(") and embedding_str.endswith(")"):
        embedding_str = embedding_str[len("array("):-1]
    try:
        return np.array(ast.literal_eval(embedding_str))
    except Exception as e:
        try:
            s = embedding_str.strip().strip("[]")
            parts = s.split(",") if "," in s else s.split()
            floats = [float(x) for x in parts if x]
            return np.array(floats)
        except Exception as e2:
            return None

# Convert the embedding strings back into NumPy arrays.
df['Combined_Embedding'] = df['Combined_Embedding'].apply(
    lambda x: convert_embedding(x) if isinstance(x, str) and x.strip() != "" else None
)
df = df[df['Combined_Embedding'].notnull()]

# Stack embeddings and compute the cosine similarity matrix.
embeddings = np.stack(df['Combined_Embedding'].values)
similarity_matrix = cosine_similarity(embeddings)

def get_similar_books(book_index, top_n=5):
    """
    Given a book index, return a list of tuples (index, similarity) for the top_n similar books.
    """
    sim_scores = list(enumerate(similarity_matrix[book_index]))
    # Skip the book itself (index 0) and return the next top_n results.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return sim_scores

# Prepare a list of books (id and title) to be injected into the HTML template.
books_list = [{"id": i, "title": title} for i, title in enumerate(df["Title"].tolist())]

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)

@app.route("/")
def index():
    # Render the index template and pass the list of books.
    return render_template("index.html", books=books_list)

@app.route("/recommend", methods=["GET"])
def recommend():
    # Retrieve the book id from the query string.
    book_id = request.args.get("book_id", type=int)
    if book_id is None or book_id < 0 or book_id >= len(df):
        return jsonify({"error": "Invalid book id"}), 400
    recs = get_similar_books(book_id, top_n=5)
    recommendations = [
        {"id": idx, "title": df.loc[idx, 'Title'], "similarity": round(score, 4)}
        for idx, score in recs
    ]
    return jsonify({"selected": df.loc[book_id, 'Title'], "recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)

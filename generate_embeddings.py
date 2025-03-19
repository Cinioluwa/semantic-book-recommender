import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load the Original Dataset
# ---------------------------
df = pd.read_csv('prod_dataset.csv')
df['Description'] = df['Description'].fillna('')

# ---------------------------
# Combine Title and Description
# ---------------------------
# Concatenate the title and description (with a period for separation)
combined_texts = (df['Title'] + ". " + df['Description']).tolist()

# ---------------------------
# Generate Embeddings
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(combined_texts, show_progress_bar=True)

# ---------------------------
# Save the New CSV File
# ---------------------------
# Convert each embedding (a NumPy array) to a string representation
df['Combined_Embedding'] = [str(list(e)) for e in embeddings]
df.to_csv('prod_dataset_combined_embeddings.csv', index=False)

print("Combined embeddings saved to prod_dataset_combined_embeddings.csv")

# Semantic Book Recommender

A Flask-based book recommendation system that leverages Sentence Transformers to generate semantic embeddings from book titles and descriptions for accurate, dynamic recommendations.

## Features

- **Combined Embeddings:** Generates embeddings by combining the book title and description for improved semantic similarity
- **Dynamic Web UI:** Modern, responsive UI using custom CSS (inspired by Bootstrap) that supports:
  - Searching for books by title
  - Browsing all available books with pagination
  - Real-time dynamic recommendations via an API endpoint
- **Flexible Deployment:** Easily packaged as a standalone executable (using PyInstaller), a Docker container, or wrapped in an Electron/Tauri desktop app
- **REST API:** Provides a `/recommend` endpoint for fetching similar book recommendations

## Requirements

- Python 3.7 or higher
- Flask
- Pandas
- NumPy
- scikit-learn
- sentence-transformers

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/book-recommender-app.git
   cd book-recommender-app
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   If you don't have a requirements.txt, you can install manually:
   ```bash
   pip install flask pandas numpy scikit-learn sentence-transformers
   ```

4. **Prepare the Dataset and Generate Embeddings:**
   - Place your original dataset as `prod_dataset.csv` in the root folder
   - Run the embedding generation script (if not done already):
     ```bash
     python generate_embeddings.py
     ```
   - This will create a new file `prod_dataset_combined_embeddings.csv` with the combined embeddings

5. **Run the Flask App:**
   ```bash
   python app.py
   ```
   - The app will be available at http://127.0.0.1:5000

## Project Structure

```
book-recommender-app/
│
├── app.py                                # Main Flask application
├── generate_embeddings.py                # Script to generate embeddings from prod_dataset.csv
├── prod_dataset.csv                      # Original dataset with book details
├── prod_dataset_combined_embeddings.csv  # Dataset with combined embeddings (generated)
├── requirements.txt                      # Python dependencies
├── templates/
│   └── index.html                        # HTML template for the web UI
└── README.md                             # This file
```

## Packaging for Distribution

### Standalone Executable (PyInstaller)

To create a standalone executable:

1. Ensure all necessary files are included using relative paths
2. Run:
   ```bash
   pyinstaller --onefile --add-data "templates:index" --add-data "prod_dataset_combined_embeddings.csv:." app.py
   ```
3. The executable will be located in the `dist/` folder

### Docker

A sample Dockerfile is provided:

```dockerfile
FROM python:3.9
WORKDIR /app

COPY app.py .
COPY templates/ templates/
COPY prod_dataset_combined_embeddings.csv .

RUN pip install flask pandas numpy scikit-learn sentence-transformers

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run with:
```bash
docker build -t book-recommender .
docker run -p 5000:5000 book-recommender
```

### Desktop App with Electron/Tauri

You can also wrap this Flask app in Electron or Tauri for a full desktop experience. See the respective documentation for further details.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [Flask](https://flask.palletsprojects.com/)
- Thanks to all the contributors and open-source projects that made this work possible!

# Text Analysis and Summarization Application

This project is a Streamlit-based application that allows users to analyze, summarize, and extract insights from text data sourced from PDFs, images, or direct text input. The app integrates various tools like OCR (Tesseract), Cohere embeddings, and text generation APIs to provide a comprehensive text processing experience.

---

## Features

- **Text Extraction:**
  - Extract text from PDF files.
  - Perform OCR on images to extract text.
  - Analyze and process plain text input.

- **Text Summarization:**
  - Generates summaries in different lengths (Short, Medium, Long).
  - Highlights key phrases using RAKE (Rapid Automatic Keyword Extraction).

- **Question-Answering:**
  - Allows users to input questions related to the uploaded text.
  - Utilizes Cohere embeddings and generation APIs to provide contextually relevant answers.

- **Advanced Options:**
  - Customize generation parameters like temperature and max tokens.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up Python Environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Install Dependencies:**
   - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract): Ensure Tesseract is installed and added to your system's PATH.
   - **Environment Variables:** Create a `.env` file and include the following:
     ```
     API_KEY=<your-cohere-api-key>
     BASE=<your-base-prompt>
     ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Select Input Type:**
   - Choose between PDF, Text, or Image input options from the dropdown.

2. **Provide Input:**
   - Upload a PDF file, image file, or paste text directly into the provided text area.

3. **Customize Summary Options:**
   - Select the desired summary length: Short, Medium, or Long.
   - Optional: Adjust temperature for the Cohere API.

4. **Ask Questions:**
   - Input a question related to the text to receive contextually relevant answers.

5. **View Results:**
   - The application will display the generated summary and highlight important key points.

---

## Code Structure

- **`app.py`**: Main application file with Streamlit implementation.
- **Functions:**
  - `extractTextFromImage`: Extracts text from an image using Tesseract OCR.
  - `highlight_key_points`: Highlights key phrases in the text.
  - `extractTextFromPdf`: Extracts text from PDF files using `pdfplumber`.
  - `processTextInput`: Processes input text into manageable chunks.
  - `convertToList`: Converts text chunks into a list of strings.
  - `embed`: Generates text embeddings using Cohere API.
  - `topNNeighbours`: Finds the top N neighbors for a given prompt embedding.
  - `generate`: Generates text using the Cohere API.

---

## Requirements

- **Python Packages:**
  - `numpy`
  - `pandas`
  - `streamlit`
  - `pdfplumber`
  - `pytesseract`
  - `rake-nltk`
  - `nltk`
  - `Pillow`
  - `cohere`
  - `python-dotenv`

- **External Tools:**
  - Tesseract OCR

---

## Notes

- Ensure that Tesseract OCR is installed and properly configured on your system.
- API key for Cohere is required for text embeddings and generation.
- Internet connectivity is needed to use the Cohere API and for NLTK stopwords download.

---

## Troubleshooting

- **Tesseract Not Found:**
  - Ensure Tesseract is installed and added to the PATH.
  - Verify the Tesseract binary path using `shutil.which("tesseract")`.

- **Cohere API Issues:**
  - Confirm your API key is valid.
  - Check the API quota and limits.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

Developed by [Your Name].

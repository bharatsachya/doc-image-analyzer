import pytesseract
import pdfplumber
import sys
from src.exception import CustomException
from PIL import Image
from src.logger import logging

def extractTextFromImage(image: Image.Image):
    logging.info("Extracting text from image")
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise CustomException(e,sys)
    

# Get the text from pdf file.
def extractTextFromPdf(pdfPath: str):
    logging.info("Extracting text from pdf")
    try:
        with pdfplumber.open(pdfPath) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise CustomException(e,sys)

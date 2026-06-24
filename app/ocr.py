import pytesseract
from PIL import Image
import fitz  # PyMuPDF

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(file_path: str) -> str:
    text = ""

    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)

        for page in doc:
            page_text = page.get_text()

            # If text exists → use it
            if page_text.strip():
                text += page_text + "\n"
            else:
                # Fallback to OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"

    else:
        img = Image.open(file_path).convert("L")
        text = pytesseract.image_to_string(img,
                                           config="--oem 3 --psm 6"
                                           )

    return text
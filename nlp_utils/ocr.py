# Import libraries
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os
from PyPDF2 import PdfFileReader

Image.LOAD_TRUNCATED_IMAGES = True
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def tesseract_ocr(PDF_file, temp_image_folder):

    # try:

    '''
    Part #1 : Converting PDF to images
    '''

    # Store all the pages of the PDF in a variable
    pages = convert_from_path(PDF_file, 500)


    # Iterate through all the pages stored above
    for image_counter, page in enumerate(pages):

        # Declaring filename for each page of PDF as JPG

        filename = "page_"+str(image_counter)+".jpg"

        # Save the image of the page in system
        page.save(os.path.join(temp_image_folder, filename), 'JPEG')


    '''
    Part #2 - Recognizing text from the images using OCR
    '''

    output_text = ''
    # Iterate from 1 to total number of pages
    for i in range(len(pages)):

        filename = os.path.join(temp_image_folder,"page_"+str(i)+".jpg")

        # Recognize the text as string in image using pytesserct
        im = Image.open(filename)
        text = str(pytesseract.image_to_string(im))
        text = text.replace('-\n', '')

        # Delete image to save space
        os.remove(filename)

        output_text += text

    return output_text


def text_extractor_pypdf2(path):

    with open(path, 'rb') as f:
        pdf = PdfFileReader(f, strict = False)
        length = pdf.getNumPages()
        doctext = ''

        #get page i, starting at 0, and then extract text from page using extractText() method
        for i in range(length):
            page = pdf.getPage(i)
            #print('Page type: {}'.format(str(type(page))))
            # doctext += page.extractText().replace("- ", "").replace("-", "") + "\n"
            doctext += page.extractText().replace("- ", "") + "\n"

    return doctext


if __name__ == '__main__':
    # Path of the pdf
    PDF_file = r"C:\Users\aspit\OneDrive\MHD NLP\Updated SEAMs\25th (Twenty-Fifth) SEAM\5th (Session V) - Status Reports National Programs\Session V [177-183].pdf"
    temp_image_folder = r'C:\Users\aspit\Git\NLP\RA_NLP_MHD\OCR\OCRlib'
    text = tesseract_ocr(PDF_file, temp_image_folder)

    print(text)

    print("Done!")

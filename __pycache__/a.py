import re
import pytesseract
def pytesseract_image_to_string(img, oem=3, psm=7) -> str:
    '''
    oem - OCR Engine Mode
        0 = Original Tesseract only.
        1 = Neural nets LSTM only.
        2 = Tesseract + LSTM.
        3 = Default, based on what is available.
    psm - Page Segmentation Mode
        0 = Orientation and script detection (OSD) only.
        1 = Automatic page segmentation with OSD.
        2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
        3 = Fully automatic page segmentation, but no OSD. (Default)
        4 = Assume a single column of text of variable sizes.
        5 = Assume a single uniform block of vertically aligned text.
        6 = Assume a single uniform block of text.
        7 = Treat the image as a single text line.
        8 = Treat the image as a single word.
        9 = Treat the image as a single word in a circle.
        10 = Treat the image as a single character.
        11 = Sparse text. Find as much text as possible in no particular order.
        12 = Sparse text with OSD.
        13 = Raw line. Treat the image as a single text line,
            bypassing hacks that are Tesseract-specific.
    '''
    tess_string = pytesseract.image_to_string(img, config=f'--oem {oem} --psm {psm}')
    regex_result = re.findall(r'[A-Z0-9]', tess_string)  # filter only uppercase alphanumeric symbols
    return ''.join(regex_result)
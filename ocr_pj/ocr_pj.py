import os
import tensorflow as tf
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# 设置Tesseract的路径
pytesseract.pytesseract.tesseract_cmd = r'D:\ProgramFiles\pylibtesseract.exe'

def convert_pdf_to_images(pdf_path):
    """pdf2image"""
    return convert_from_path(pdf_path)

def convert_image_to_text(image):
    """image2txt_ocr"""
    return pytesseract.image_to_string(image)

def extract_book_title(image_path, unwanted_chars):
    """find the name of the book"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image).strip()
    
    # delete unnecessary words
    for char in unwanted_chars:
        text = text.replace(char, '')
    
    return text.strip() if text else None  # book's name(after processing)/none

def create_tfrecord(output_path, text_data):
    """txt2TFR"""
    with tf.io.TFRecordWriter(output_path) as writer:
        for text in text_data:
            if text:
                feature = {
                    'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8')])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def process_pdf_to_tfrecord(pdf_path, output_tfrecord, unwanted_chars):
    """bookname2TFR_ocr"""
    text_data = []
    images = convert_pdf_to_images(pdf_path)
    for index, image in enumerate(images):
        # bookname
        book_title = extract_book_title(image, unwanted_chars)
        if book_title:
            print(f"提取到书籍名字（第{index+1}页）: {book_title}")
            text_data.append(book_title)
        
        # pagecontent_ocr
        page_text = convert_image_to_text(image)
        if page_text:
            text_data.append(page_text)

    create_tfrecord(output_tfrecord, text_data)

if __name__ == "__main__":
    pdf_path = '你的PDF文件路径.pdf'  # ur_pdf_path
    output_tfrecord = 'output.tfrecord'  # TFRecordName
    unwanted_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']  # words2delete

    process_pdf_to_tfrecord(pdf_path, output_tfrecord, unwanted_chars)
    print("TFRecord文件创建完成！")

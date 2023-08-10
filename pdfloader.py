from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import cv2
from math import floor, ceil
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import arabic_reshaper 
from bidi.algorithm import get_display

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

from glob import glob 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
import argparse

'''
This function takes in a pdf file path processes it using the Azure Form Recognizer custom model
and returns the result containing the fields and their values
'''
def extract_fields_using_form_rec(pdf_path):
    endpoint = os.environ.get("FR_ENDPOINT")
    key = os.environ.get("FR_KEY")

    model_id = os.environ.get("FR_MODEL_ID")

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    with open(pdf_path, "rb") as f:
    # Make sure your document's type is included in the list of document types the custom model can analyze
        poller = document_analysis_client.begin_analyze_document(model_id, document=f)
    result = poller.result()
    return result

'''
This function takes in a pdf file path and converts it into images, easier to work with images than pdfs
'''
def convert_pdf_to_images(pdf_path, poppler_path):
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    # Loop through each image
    for i, image in enumerate(images):
        # Save the image
        image.save("./images/" + str(i) + '.png', "PNG")

'''
Translate the text using Azure Translator
'''
def translate_text(text, source_language="en", target_language="german"):
    endpoint = os.environ.get("TRANSLATOR_ENDPOINT")
    key = os.environ.get("TRANSLATOR_KEY")
    ta_credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    result = client.translate(text, target_language=target_language, source_language=source_language)
    return result

'''Translate the text using OpenAI'''
def translate_text_using_openai(text, source_language="en", target_language="german"):
    # Set the ENV variables that Langchain needs to connect to Azure OpenAI
    os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
    os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
    os.environ["OPENAI_API_TYPE"] = "azure"
    MODEL = os.environ["AZURE_OPENAI_MODEL_ID"]

    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=300)

    # Now we create a simple prompt template
    prompt = PromptTemplate(
        input_variables=["source","text_to_translate", "target"],
        template='Translate the following from {source} to {target}, if it is an entity like Name or Country do not translate and return the text as is: "{text_to_translate}". Give your response in {target}',
    )

    # And finally we create our first generic chain
    chain_chat = LLMChain(llm=llm, prompt=prompt)
    response = chain_chat({"source": source_language, "target": target_language, "text_to_translate": text})
    return response['text']


def translate_pdf(result, translated_pdf_path, source_language="en", target_language="german"):
    for idx, document in enumerate(result.documents):

        # Load the image
        img = cv2.imread('./images/'+str(idx)+'.png')
        height_pixels, width_pixels, c = img.shape #height, width and channel of the image in pixels
        
        width_inches, height_inches = document.bounding_regions[0].polygon[2] #width and height of the image in inches returned by the form recognizer
        for name, field in document.fields.items():
            top_left_x_inches, top_left_y_inches = field.bounding_regions[0].polygon[0] # top left x and y coordinates of the field in inches returned by the form recognizer
            bottom_right_x_inches, bottom_right_y_inches = field.bounding_regions[0].polygon[2] # bottom right x and y coordinates of the field in inches returned by the form recognizer
            
            field_value = field.value if field.value else field.content #text to convert

            translated_text = translate_text_using_openai(field_value, source_language, target_language) #translate the text
    
            # Specify the coordinates for the redaction rectangle
            top_left_x = floor((top_left_x_inches/width_inches) * width_pixels) - 5 #(coordinate in inches/side in inches)*side in pixel
            top_left_y = floor((top_left_y_inches/height_inches) * height_pixels)
            bottom_right_x = ceil((bottom_right_x_inches/width_inches) * width_pixels) + 5
            bottom_right_y = ceil((bottom_right_y_inches/height_inches) * height_pixels)
            x, y, width, height = top_left_x, top_left_y, (bottom_right_x - top_left_x), (bottom_right_y - top_left_y)
    
            # Create a grey rectangle to cover the desired portion of the image
            color = (219, 219, 219)
            img[y:y + height, x:x + width] = color
    
            # # Write text on the red rectangle using a white color
            # font = cv2.FONT_HERSHEY_PLAIN#
            # font = ImageFont.truetype("arial.ttc", 32)
            # #org = (x + int(width / 4), y + int(height / 2))
            # org = (x, y + int(height / 2) + 5)
            # fontScale = 0.7
            # color = (0, 0, 0)
            # thickness = 1
            # text = translated_text.replace('"', '')
            # img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            # ## Use simsum.ttc to write Chinese.
            fontpath = "arial.ttf" #
            font = ImageFont.truetype(fontpath, 28)
            img_pil = Image.fromarray(img)
            # draw = ImageDraw.Draw(img_pil)
            #reshaped_text = arabic_reshaper.reshape(text)
            #bidi_text = get_display(reshaped_text) 
            draw = ImageDraw.Draw(img_pil)
            draw.text((x,y),  translated_text.replace('"', ''), font = font, fill = (0,0,0,0))
            img = np.array(img_pil)
        # Save the resulting image
        cv2.imwrite('./redacted_images/'+str(idx)+'.png', img)

    iml = []
    files = glob("./redacted_images/*.png")
    # print(f"{files=}")
    for img in files:
        imgs = Image.open(img)
        iml.append(imgs)
    #pdf = "redacted_pdf_file.pdf"
    # print(iml)
    image = iml[0]
    iml.pop(0)
    image.save(translated_pdf_path, "PDF" , resolution=100.0, save_all=True, append_images=iml)

def main(pdf_path, out_path):
    load_dotenv()
    poppler_path = os.environ.get("POPPLER_PATH")
    source_language = os.environ.get("SOURCE_LANGUAGE")
    target_language = os.environ.get("TARGET_LANGUAGE")
    convert_pdf_to_images(pdf_path, poppler_path)
    result_field = extract_fields_using_form_rec(pdf_path)
    translate_pdf(result_field, out_path, source_language, target_language)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_pdf")
    parser.add_argument("output_pdf")
    args = parser.parse_args()
    # print(args)
    main(args.input_pdf, args.output_pdf)

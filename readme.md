# PDF Translator
The PDF Translator utility is used to translate a subset of the PDF document from one language to another. There are scenarios where you'd want to translate items within an invoice and keep the invoice header and structure as-is. This code focuses on the translation peice, you can modify the code to make sure the PDF looks good.
The current codebase uses Azure OpenAI to perform translation, you can use Azure Translator (code already provided, you would need to modify the codebase to call the function)

# Prerequisites
- Azure subscription
- Azure OpenAI service with gpt-35-turbo model deployed
- Azure Forms Recogniser service
- Install Poppler on your machine (this is a required dependency, without which the code won't function)

# Step 1 - Create Forms Recogniser model
1. You can either use an existing Forms Recogniser model or create a custom model. Make sure you train the custom model on the documents you are going to process and identify the fields you wish to be converted.
2. Save the model

# Step 2 - Code
1. Install the required libraries  from requirements.txt
2. Copy the sample_env file and rename it to ".env"
3. Populate the ".env" file with values
4. Execute the pdfloader.py <input_pdf> <output_pdf>

Attached screenshots of the translation:

Original
![Original PDF](original.png)
German
![German Translation](german.png)
Spanish
![Spanish Translation](spanish.png)

# Next steps - TODO
1. Get color, font from a config file. Currently all translated/replaced text have a standard light grey color and written in black. Next step would be to read these from a config file, e.g.
```json
{
    'fields:[
        'field_name':{
            'font': 'arial.ttc',
            'font_size': '20',
            'font_color': '(219,219,219)'
            'background_color': '(255,255,255)'
        }
    ]
}
```
2. Ability to select translation service, OpenAI or Translator
import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.cuda.amp import autocast

# Load the saved model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('./saved_T5_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_T5_model')

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def summarize_input(input_data, input_type='pdf'):
    # Function to read PDF and extract text
    def read_pdf(file_path):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    # Function to clean text
    def clean_text(text):
        # Remove forward tags and other unwanted patterns
        text = re.sub(r'Fwd:|Forwarded message:', '', text, flags=re.IGNORECASE)
        text = text.replace('\n', ' ').replace('\r', '')
        return text

    # Read and clean the input content
    if input_type == 'pdf':
        input_text = read_pdf(input_data)
    elif input_type == 'text':
        input_text = input_data
    else:
        raise ValueError("Invalid input type. Use 'pdf' or 'text'.")

    cleaned_text = clean_text(input_text)

    # Tokenize and generate summary
    inputs = tokenizer(cleaned_text, return_tensors='pt', max_length=1024, truncation=True, padding='max_length').to(device)
    with autocast():
        summary_ids = model.generate(inputs['input_ids'], max_length=512, min_length=40, length_penalty=2.0, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example usage
pdf_summary = summarize_input('Email_thread.pdf', input_type='pdf')
print(pdf_summary)
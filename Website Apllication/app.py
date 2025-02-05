from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the summarization model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('./saved_T5_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_T5_model')

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def clean_text(text):
    """Clean email thread text."""
    text = re.sub(r'Fwd:|Forwarded message:', '', text, flags=re.IGNORECASE)
    return text.replace('\n', ' ').replace('\r', '')

def summarize_text(input_text):
    """Summarize the given text."""
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True, padding='max_length').to(device)
    summary_ids = model.generate(
        inputs['input_ids'], max_length=512, min_length=40, length_penalty=2.0, num_beams=2, early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Check if a file is provided
        if 'file' in request.files and request.files['file']:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                doc = fitz.open(stream=file.read(), filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                return jsonify({'error': 'Unsupported file format. Use PDF or TXT.'}), 400
        else:
            text = request.form.get('email_text', '')
        
        # Clean and summarize text
        cleaned_text = clean_text(text)
        if not cleaned_text.strip():
            return jsonify({'error': 'No valid text provided for summarization.'}), 400

        summary = summarize_text(cleaned_text)
        return jsonify({'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

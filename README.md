# Email Thread Summarization

This repository contains the implementation of an email thread summarization tool that leverages transformer-based models like T5 and Pegasus to generate concise, informative summaries of lengthy email conversations. By preserving the context and intent of the original messages, the tool improves information accessibility and reduces cognitive load for users.

## Features
- Summarization using a two-layer T5 model.
- Preprocessing pipeline for cleaning and structuring raw email data.
- Fine-tuned T5 model trained on the *Email Thread Summary* dataset.
- Evaluation metrics, including ROUGE, G-Eval, and SummEval, for assessing summary quality.
- Multilingual translation of generated summaries for global accessibility.

## Usage
### Run the Application
To start the web application for generating summaries, execute:
```bash
python Website Application/app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.

## Dataset
We use the *Email Thread Summary* dataset from Hugging Face: https://huggingface.co/datasets/sidhq/email-thread-summary

## Citation
If you use this repository, please cite:
```bibtex
@misc{EmailThreadSummarization2024,
  author = {Srikitha Kandra, Venkata Tejaswi Kalla, Tejaswi Samineni},
  title = {Email Thread Summarization},
  year = {2024},
  url = {https://github.com/SrikithaKandra/EmailThreadSummarization}
}
```

## Acknowledgments
- **Hugging Face Transformers** for providing pre-trained models and tokenizers.
- The creators of the *Email Thread Summary* dataset.
- Libraries such as PyTorch, Optuna, and Scikit-learn for their essential tools.

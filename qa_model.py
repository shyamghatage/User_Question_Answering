# qa_model.py
from transformers import pipeline

# Load the pre-trained QA pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

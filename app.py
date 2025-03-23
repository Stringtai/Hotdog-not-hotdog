import gradio as gr
from transformers import pipeline

# Makes the image classification pipeline with the model
model = pipeline(task="classification", model='julien-c/hotdog-not-hotdog')

# Defines the prediction function
def predict(image):
    predictions = model(image)
    return {p['label']: p['score'] for p in predictions}

# Creates the Gradio interface
gr.Interface(
    predict,
    inputs=gr.Image(label='Upload Image', type='filepath'),
    outputs=gr.Label(num_top_classes=2),
    title='Hotdog or Not Hotdog?'
).launch() # Lauches the interface to Hugging Face

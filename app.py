import gradio as gr
from transformers import pipeline

model = pipeline(task="classification", model='julien-c/hotdog-not-hotdog')

def predict(image):
    predictions = model(image)
    return {p['label']: p['score'] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.Image(label='Upload Image', type='filepath'),
    outputs=gr.Label(num_top_classes=2),
    title='Hotdog or Not Hotdog?'
).launch()
import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from train import LabelConverter

# Load the session
session = ort.InferenceSession("/Users/nikhil/Desktop/Leegality_task/ocr_model.onnx", providers=['CPUExecutionProvider'])

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((64,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    converter = LabelConverter()
    return img_tensor, converter

def predict_onnx(image, max_len=32):
    # Start with <SOS>
    img_tensor, converter = preprocess(image)
    input_ids = np.array([[converter.SOS]], dtype=np.int64)
    image_np = img_tensor.numpy()

    for _ in range(max_len):
        # Run inference
        outputs = session.run(None, {
            'images': image_np,
            'tgt': input_ids
        })
        
        # Get last predicted character
        logits = outputs[0]
        next_token = np.argmax(logits[0, -1, :])
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

        if next_token == converter.EOS:
            break
            
    return converter.decode(input_ids[0])

# 3. Create Gradio Interface
interface = gr.Interface(
    fn=predict_onnx,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="IIIT5K Word Recognition OCR",
    description="Upload a cropped word image to see the model's prediction."
)

interface.launch()
import onnxruntime as ort
import numpy as np
from PIL import Image
import numpy as np
from train import LabelConverter
import pandas as pd
import torch
from torchvision import transforms

# Load the session
session = ort.InferenceSession("/Users/nikhil/Desktop/Leegality_task/ocr_model.onnx", providers=['CPUExecutionProvider'])

def predict_onnx(image, converter, max_len=32):
    # Start with <SOS>
    input_ids = np.array([[converter.SOS]], dtype=np.int64)
    image_np = image.numpy()

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

def main():
    img = Image.open("/Users/nikhil/Desktop/Leegality_task/archive/IIIT5K-Word_V3.0/IIIT5K/test/6_10.png").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    train_df = pd.read_csv("/Users/nikhil/Desktop/Leegality_task/archive/traindata.csv")
    all_text = "".join(train_df['GroundTruth'].astype(str).tolist())
    converter = LabelConverter(all_text)
    prediction = predict_onnx(img_tensor, converter)
    print(prediction)   

if __name__ == "__main__":
    main()  

    
    

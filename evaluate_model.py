import onnxruntime as ort
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from Levenshtein import distance
from tqdm import tqdm
from train import LabelConverter

# 1. Configuration
TEST_CSV = 'archive/testdata.csv'  # Path to your test CSV
TEST_DIR = 'archive/IIIT5K-Word_V3.0/IIIT5K/'         # Folder containing test images
ONNX_MODEL_PATH = 'ocr_model.onnx'
MAX_LEN = 32

# 2. Setup ONNX Session
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0).numpy()

def greedy_decode_onnx(image_np, converter):
    # Start with <SOS> token
    input_ids = np.array([[converter.SOS]], dtype=np.int64)
    
    for _ in range(MAX_LEN):
        # Run ONNX inference
        outputs = session.run(None, {
            'images': image_np,
            'tgt': input_ids
        })
        
        # Get the character with the highest probability for the LAST position
        logits = outputs[0]
        next_token = np.argmax(logits[0, -1, :])
        
        # Append and check for <EOS>
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
        if next_token == converter.EOS:
            break
            
    return converter.decode(input_ids[0])

def evaluate(converter):
    test_df = pd.read_csv(TEST_CSV)
    abs_correct = 0
    lex_correct = 0
    total = len(test_df)
    
    print(f"Starting evaluation on {total} images...")
    
    for _, row in tqdm(test_df.iterrows(), total=total):
        img_path = TEST_DIR + row['ImgName']
        ground_truth = str(row['GroundTruth']).upper()
        
        # 1. Get raw model prediction
        image_np = preprocess_image(img_path)
        raw_pred = greedy_decode_onnx(image_np, converter).upper()
        
        # 2. Absolute Accuracy
        if raw_pred == ground_truth:
            abs_correct += 1
            
        # 3. Small Lexicon Accuracy (Top-50 candidates)
        # Assumes small lexicon is comma-separated in the CSV
        small_lexicon = str(row['smallLexi']).upper().split(',')
        
        # Use Levenshtein distance to find the best match in the candidate list
        best_match = min(small_lexicon, key=lambda x: distance(raw_pred, x))
        #print(str(best_match[2:-1]), ground_truth, raw_pred)
        if str(best_match[2:-1]) == ground_truth:
            lex_correct += 1

    # Output final metrics
    print(f"\n--- Evaluation Results ---")
    print(f"Absolute Accuracy: {abs_correct/total*100:.2f}%")
    print(f"Small Lexicon Accuracy (50 words): {lex_correct/total*100:.2f}%")

if __name__ == "__main__":
    # Ensure you initialize the same converter used during training
    train_df = pd.read_csv("/Users/nikhil/Desktop/Leegality_task/archive/traindata.csv")
    all_text = "".join(train_df['GroundTruth'].astype(str).tolist())
    converter = LabelConverter(all_text)
    evaluate(converter)
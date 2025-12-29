import torch
import torch.ao.quantization as quant
from model_q import OCRModel
from train import IIIT5KDataset, collate_fn, LabelConverter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import json
import pandas as pd

# 1. SETUP ENVIRONMENT
torch.backends.quantized.engine = 'qnnpack'
device = torch.device('cpu')

# Load Config
with open('config.json', 'r') as f:
    config = json.load(f)

# Build Converter to match training
train_df = pd.read_csv(config['data']['train_csv'])
all_text = "".join(train_df['GroundTruth'].astype(str).tolist())
converter = LabelConverter(all_text)

# 2. INITIALIZE ARCHITECTURE
model = OCRModel(
    vocab_size=converter.vocab_size,
    d_model=config['model']['d_model'],
    nhead=config['model']['nhead'],
    num_encoder_layers=config['model']['num_encoder_layers'],
    num_decoder_layers=config['model']['num_decoder_layers'],
    resnet_layers=config['model']['resnet_layers'],
    max_len=config['data']['max_len']
).to(device)

# Load the FP32 best weights before quantization starts
model.load_state_dict(torch.load('checkpoints/ocr_model_fold0_best.pth', map_location=device))
model.eval()

# 3. CONFIGURE QUANTIZATION
model.qconfig = quant.get_default_qconfig('qnnpack')
# Mandatory: Use weight-only config for the embedding layer
model.embedding.qconfig = quant.float_qparams_weight_only_qconfig

# 4. PREPARE & CALIBRATE
# This step adds 'observers' to the model to learn the data distribution
model_prepared = quant.prepare(model, inplace=False)

preprocess = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

calibration_dataset = IIIT5KDataset(
    csv_file=config['data']['train_csv'],
    root_dir=config['data']['train_dir'],
    size=(64, 256),
    transform=preprocess,
    converter=converter
)
calibration_loader = DataLoader(calibration_dataset, batch_size=1, sampler=SubsetRandomSampler(range(100)), collate_fn=collate_fn)

# Run calibration
with torch.no_grad():
    for images, _ in calibration_loader:
        # Use a dummy target length matching your config
        dummy_tgt = torch.zeros((1, config['data']['max_len']), dtype=torch.long)
        model_prepared(images, dummy_tgt)

# 5. CONVERT & SAVE
# This swaps the layers for their INT8 versions
model_quantized = quant.convert(model_prepared, inplace=False)
torch.save(model_quantized.state_dict(), "model_quantized.pth")
print("INT8 Quantized model saved as model_quantized.pth")
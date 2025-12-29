import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import collections
from model import OCRModel
from straug.geometry import Perspective, Rotate
from straug.blur import MotionBlur, GaussianBlur
from straug.noise import GaussianNoise, ShotNoise
from straug.camera import Contrast, Brightness
import random

# Vocabulary Handling
class LabelConverter:
    def __init__(self, chars):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.char2idx = {c: i+4 for i, c in enumerate(self.chars)}
        self.idx2char = {i+4: c for i, c in enumerate(self.chars)}
        
        # Special tokens
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3
        
        self.char2idx['<PAD>'] = self.PAD
        self.char2idx['<SOS>'] = self.SOS
        self.char2idx['<EOS>'] = self.EOS
        self.char2idx['<UNK>'] = self.UNK
        
        self.idx2char[self.PAD] = '<PAD>'
        self.idx2char[self.SOS] = '<SOS>'
        self.idx2char[self.EOS] = '<EOS>'
        self.idx2char[self.UNK] = '<UNK>'
        
        self.vocab_size = len(self.char2idx)

    def encode(self, text):
        indices = [self.SOS]
        for char in text:
            indices.append(self.char2idx.get(char, self.UNK))
        indices.append(self.EOS)
        return torch.LongTensor(indices)

    def decode(self, indices):
        chars = []
        for idx in indices:
            idx = idx.item()
            if idx == self.EOS:
                break
            if idx == self.SOS or idx == self.PAD:
                continue
            chars.append(self.idx2char.get(idx, '<UNK>'))
        return "".join(chars)

def resize_and_pad_aspect(img, target_size):
    w, h = img.size
    th, tw = target_size
    # Resize to fit within target, maintaining aspect ratio
    if h > w: # Height is longer
        new_h = th
        new_w = int(w * th / h)
    else: # Width is longer or equal
        new_w = tw
        new_h = int(h * tw / w)

    resized_img = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
    
    # Now pad to make it square (TARGET_SIZE, TARGET_SIZE)
    pad_w = (tw - new_w) // 2
    pad_h = (th - new_h) // 2

    out_img = np.zeros((th, tw, 3), dtype=np.uint8)
    out_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = np.array(resized_img)
    
    return Image.fromarray(out_img)

class IIIT5KDataset(Dataset):
    def __init__(self, csv_file, root_dir, size=(64,256), transform=None, converter=None, use_straug=False):
        self.df = pd.read_csv(csv_file)
        self.size = size
        self.root_dir = root_dir
        self.transform = transform
        self.converter = converter
        self.use_straug = use_straug # Initially False
        
        self.augmentors = [
            Perspective(),
            Rotate(),
            MotionBlur(),
            GaussianNoise(),
            Contrast(),
            Brightness()
        ]

    def __len__(self):
        return len(self.df)

    def set_aug(self, mode: bool):
        self.use_straug = mode

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0] # ImgName column
        label = self.df.iloc[idx, 1]    # GroundTruth column
        
        # Construct full image path
        # Assuming ImgName contains 'train/...' or 'test/...'
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            # image = resize_and_pad_aspect(image, self.size)
        except FileNotFoundError:
            # Handle missing file if any, or just fail loud
            print(f"Warning: Image not found at {img_path}")
            # Return a dummy item or raise error. 
            # For simplicity, let's create a black image
            image = Image.new('RGB', self.size)

        if self.use_straug:
            aug = self.augmentors[np.random.randint(0, len(self.augmentors))]
            image = aug(image, mag=1)
            
        if self.transform:
            image = self.transform(image)
            
        target = self.converter.encode(str(label))
        
        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    
    # Pad targets
    lengths = [len(t) for t in targets]
    max_length = max(lengths)
    
    padded_targets = torch.full((len(targets), max_length), 0, dtype=torch.long) # 0 is PAD
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t
        
    return images, padded_targets

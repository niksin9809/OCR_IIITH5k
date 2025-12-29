import torch
import torch.nn as nn
import torchvision.models as models
import math

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet_layers=18, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        if resnet_layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif resnet_layers == 34:
            resnet = models.resnet34(pretrained=pretrained)
        elif resnet_layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"ResNet layers {resnet_layers} not supported")

        if resnet_layers in [18, 34]:
            self.out_channels = 512
        else:
            self.out_channels = 2048

        # Standard ResNet backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        # output: [B, C, H', W']
        x = self.backbone(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class OCRModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2, resnet_layers=18, max_len=1000, dropout=0.1):
        super(OCRModel, self).__init__()
        
        # Feature Extractor
        self.feature_extractor = ResNetFeatureExtractor(resnet_layers=resnet_layers)
        
        # Adapter
        self.adapter = nn.Conv2d(self.feature_extractor.out_channels, d_model, 1)
        
        # Positional Encoding (Non-Learnable)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Final output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, tgt, tgt_mask=None, tgt_padding_mask=None):
        # images: [B, 3, H, W]
        # tgt: [B, T]
        
        # Extract features
        features = self.feature_extractor(images) # [B, 512, H', W']
        features = self.adapter(features)         # [B, d_model, H', W']
        
        # Collapse Height dimension using Max Pooling
        # features: [B, d_model, H, W] -> [B, d_model, W]
        features = torch.max(features, dim=2)[0]
        
        # Permute to [W, B, d_model] for Transformer (Seq, Batch, Fet)
        src = features.permute(2, 0, 1)
        
        # Apply 1D PE to Src
        src = self.pos_encoder(src)
        
        # Prepare target
        # tgt_emb: [T, B, d_model]
        tgt_emb = self.embedding(tgt).permute(1, 0, 2)
        
        # Apply 1D PE to Tgt
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Transformer
        output = self.transformer(
            src, 
            tgt_emb, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Output
        # output: [T, B, d_model]
        prediction = self.fc_out(output) # [T, B, vocab_size]
        
        # Return [B, T, vocab_size] for easier loss calculation
        return prediction.permute(1, 0, 2)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
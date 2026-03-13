import torch
import torch.nn as nn



class WeightedRMSELoss(nn.Module):
    def __init__(self, metric_weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(metric_weights, dtype=torch.float32))
    
    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2, dim=0)
        rmse = torch.sqrt(mse + 1e-8)
        weighted_rmse = self.weights * rmse
        return torch.sum(weighted_rmse) / torch.sum(self.weights)

class ProteinTransformerRegressorV2(nn.Module):
    def __init__(
        self, 
        n_input=4253, 
        d_model=256, 
        nhead=8, 
        num_layers=3,
        n_outputs=4, 
        max_length=100,
        dropout=[0.4, 0.1],
    ):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(n_input, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout[0])
        )
        
        pos_encoding = torch.zeros(1, max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout[1],
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_weights = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
            nn.LayerNorm(d_model // 2)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model // 2, 1) for _ in range(n_outputs)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len, n_input=4253)
        batch_size, seq_len, _ = x.shape
        
        if lengths is not None and not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=x.device)
        
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) < lengths.unsqueeze(1)
        else:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        src_key_padding_mask = ~mask
 
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        scores = self.attention_weights(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4) 
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        shared = self.shared_head(pooled)
        outputs = torch.cat([head(shared) for head in self.task_heads], dim=1)
        
        return torch.sigmoid(outputs)
import torch
import torch.nn as nn

class WaveBlock(nn.Module):
    def __init__(self, in_features, filters, kernel_size, n):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.n = n
    
        self.cas_conv1 = nn.Conv1d(in_features, filters, 1)

        dilation_rates = [2**i for i in range(n)]
        self.tanh_out_layers = nn.ModuleList([])
        self.sig_out_layers = nn.ModuleList([])
        self.cas_conv_layers = nn.ModuleList([])

        for dilation_rate in dilation_rates:
            tanh_out = nn.Sequential(*[nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate, padding='same'), nn.Tanh()])
            self.tanh_out_layers.append(tanh_out)
            sig_out = nn.Sequential(*[nn.Conv1d(filters, filters, kernel_size, dilation=dilation_rate, padding='same'), nn.Sigmoid()])
            self.sig_out_layers.append(sig_out)
            self.cas_conv_layers.append(nn.Conv1d(filters, filters, 1))
        
    def forward(self, x):
        x = self.cas_conv1(x)
        res_x = x
        
        for tanh_layer, sig_layer, conv_layer in zip(self.tanh_out_layers, self.sig_out_layers, self.cas_conv_layers):
            x = tanh_layer(x) * sig_layer(x)
            x = conv_layer(x)
        x = x + res_x
        del res_x
        return x
    
class BirdNet(nn.Module):
    def __init__(self, temporal_fearture_size=64, kernel_size=3, hidden_size=256, num_classes=264):
        super().__init__()
        self.representation_block = nn.Sequential(*[
            WaveBlock(1, 8, kernel_size, 16),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            WaveBlock(8, 16, kernel_size, 8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            WaveBlock(16, 32, kernel_size, 4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            WaveBlock(32, temporal_fearture_size, kernel_size, 1)
        ])
        self.temporal_block = nn.LSTM(temporal_fearture_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = torch.concat([self.representation_block(x[:, i, :].unsqueeze(1)).unsqueeze(0) for i in range(x.shape[1])])
        x = torch.mean(x, dim=-1)
        x = torch.sum(self.temporal_block(x)[0], axis=0)
        x = nn.ReLU()(x)
        x = self.classifier(x)
        x = nn.Sigmoid()(x)
        return x
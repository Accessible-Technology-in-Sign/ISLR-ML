import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(SimpleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=128 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    

class ComplexLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(SimpleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=128 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    

class DoubleLSTM(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        super(DoubleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size= input_features * num_coords,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=256 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(256 * 2, num_signs)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    


class TransformerClassifier(nn.Module):
    def __init__(self, *args):
        model_dim = 42  # Embedding and model dimension
        n_heads = 7  # Number of attention heads
        num_encoder_layers = 2  # Number of transformer layers
        num_classes = 563  # Number of output classes
        max_seq_length = 60  # Maximum sequence length
        super(TransformerClassifier, self).__init__()
        
        # self.embedding = nn.Embedding(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.randn(max_seq_length, model_dim))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads),
            num_layers=num_encoder_layers
        )
        
        self.fc = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        # Input shape (batch_size, seq_length)
        
        # Embed the input tokens and add positional encodings
        x = x + self.position_encoding[:x.size(1), :]
        
        # Reshape to (seq_length, batch_size, model_dim) for transformer input
        x = x.permute(1, 0, 2)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        
        # Get the representation of the last token (or alternatively the mean across tokens)
        x = x[-1, :, :]  # Shape: (batch_size, model_dim)
        
        # Pass through the final classification layer
        logits = self.fc(x)
        
        return logits
    
class ComplexTransformerClassifier(nn.Module):
    def __init__(self, *args):
        super(ComplexTransformerClassifier, self).__init__()
        model_dim = 42  # Embedding and model dimension
        n_heads = 7  # Number of attention heads
        num_encoder_layers = 5  # Number of transformer layers
        num_classes = 563  # Number of output classes
        max_seq_length = 60  # Maximum sequence length
        
        # Embedding and positional encoding
        # self.embedding = nn.Embedding(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.randn(max_seq_length, model_dim))

        # Efficient transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 2),
            num_layers=num_encoder_layers
        )
        
        # Reduce the sequence by pooling
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence length
        
        # Final classification layer
        self.fc = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        x = x + self.position_encoding[:x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer requires (seq_len, batch, model_dim)
        x = self.transformer_encoder(x)
        
        # Pool across the sequence dimension (seq_len -> 1)
        x = self.pool(x.permute(1, 2, 0)).squeeze(-1)  # Now (batch_size, model_dim)
        
        # Classification layer
        logits = self.fc(x)
        return logits
    
    
if __name__ == '__main__':
    print("Running lstm.py")
    model = TransformerClassifier()
    print(model)
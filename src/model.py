import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    Optimized 2-layer LSTM architecture with regularization
    specifically designed to avoid overfitting on C-MAPSS FD001.
    """
    def __init__(self, input_dim=14, hidden_dim_1=64, hidden_dim_2=32, fc_dim=16, dropout_prob=0.2):
        super(LSTMRegressor, self).__init__()
        
        # First LSTM layer (returns sequences)
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second LSTM layer (takes output of first LSTM)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim_1,
            hidden_size=hidden_dim_2,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Fully connected regressor head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim_2, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        
        out2, _ = self.lstm2(out1)
        # Take the output at the final timestep
        last_step = out2[:, -1, :]
        last_step = self.dropout2(last_step)
        
        rul_pred = self.fc(last_step)
        return rul_pred.squeeze(-1)


class OverfittedLSTMRegressor(nn.Module):
    """
    Overparameterized 3-layer LSTM without dropout,
    used to demonstrate the initial overfitting behavior discussed in the project abstract.
    """
    def __init__(self, input_dim=14, hidden_dim_1=256, hidden_dim_2=128, hidden_dim_3=64):
        super(OverfittedLSTMRegressor, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_1, hidden_size=hidden_dim_2, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=hidden_dim_2, hidden_size=hidden_dim_3, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim_3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(out2)
        last_step = out3[:, -1, :]
        rul_pred = self.fc(last_step)
        return rul_pred.squeeze(-1)

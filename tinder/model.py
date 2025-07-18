import torch
import torch.nn as nn

# ----------------------------
# 🔁 ConvLSTM Cell Definition
# ----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, h_prev, c_prev):
        combined = torch.cat([input_tensor, h_prev], dim=1)
        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

# ----------------------------
# 🔁 ConvLSTM Layer (multi-step)
# ----------------------------
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, input_sequence):
        batch_size, seq_len, channels, height, width = input_sequence.shape

        h, c = self._init_hidden(batch_size, height, width)

        for t in range(seq_len):
            h, c = self.cell(input_sequence[:, t], h, c)

        return h  # Final hidden state used for output

    def _init_hidden(self, batch_size, height, width):
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.cell.hidden_channels, height, width).to(device)
        c = torch.zeros(batch_size, self.cell.hidden_channels, height, width).to(device)
        return h, c

# ----------------------------
# 🔥 Fire Spread Prediction Head
# ----------------------------
class FireSpreadPredictor(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels=1):
        super(FireSpreadPredictor, self).__init__()
        self.convlstm = ConvLSTM(input_channels, hidden_channels)
        self.output_layer = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, input_sequence):
        features = self.convlstm(input_sequence)
        output = torch.sigmoid(self.output_layer(features))  # [0–1] fire probability
        return output

class FireCastTINDERModel():
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path

    def load_model(self):
        # Placeholder for loading the model
        print(f"Loading model {self.model_name} from {self.model_path}")

    def predict(self, data):
        # Placeholder for prediction logic
        print(f"Predicting with model {self.model_name} on data: {data}")
        return "Prediction result"
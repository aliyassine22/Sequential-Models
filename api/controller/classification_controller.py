import torch
import torch.nn as nn

device='cuda' if torch.cuda.is_available() else 'cpu'
class LSTM_V2(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers, num_classes):
        super(LSTM_V2,self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.lstm= nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.3,bidirectional=True) # batch must be in first dimension
        # if batch first was set to true, input shape: batch size,sequence number, features size
        self.fc=nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        num_directions = 2  # we need to accomodate the hidden and cell states to both directions
        h0=torch.zeros(self.num_layers*num_directions, x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers*num_directions, x.size(0),self.hidden_size).to(device)
        out, _= self.lstm(x,(h0,c0)) # we do not need the second returned output (h_n)
        out=out[:,-1,:]
        out = self.fc(out)
        return out
class RNNClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()

        self.bidirectional = True  # enable bidirectionality
        self.rnn = nn.GRU(
            input_size=embedding_size,   # size of each word vector
            hidden_size=hidden_size,     # size of hidden layer
            num_layers=num_layers,       # how many GRU layers
            batch_first=True,            # input shape = (batch, sequence, features)
            bidirectional=self.bidirectional  # read both directions
        )
        # Output size becomes hidden_size * 2 if bidirectional
        self.fc = nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # shape: (batch, seq_len, hidden*2 if bidirectional)
        out = out[:, -1, :]   # get the last time step output
        out = self.fc(out)    # final prediction layer
        return out
    
    
def load_model(model_type: str):
    if model_type == 'GRU':
        PATH=r"pretrained models\Recurrent Models\GRU_Best_Model"
        model = RNNClassifier(300, hidden_size=128, num_layers=1, num_classes=4)
        model.load_state_dict(torch.load(PATH, weights_only=True))

    elif model_type == 'LSTM':
        PATH=r"pretrained models\Recurrent Models\LSTM_V3_Best_Model"
        model=LSTM_V2(input_size=300,hidden_size=256,num_layers=3,num_classes=4)
        model.load_state_dict(torch.load(PATH))
    return model

def evaluate_text(model,feed_data):
    model.eval()
    with torch.no_grad():
        for input in feed_data:
            outputs=model(input)
            _, predicted = torch.max(outputs.data, 1)
            if predicted == 0:
                return 'Normal'
            elif predicted == 1:
                return 'Depression'
            elif predicted == 2:
                return 'Suicidal'
            else: 
                return 'Some Other Disorder'
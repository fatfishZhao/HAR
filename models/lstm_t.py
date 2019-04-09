import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, num_classes):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=6,
            hidden_size=36,
            num_layers=1,
            batch_first=True,
            dropout=0.5
        )
        self.out = nn.Linear(36, num_classes)
    def forward(self, x):
        # x: batchsize * length * 6
        r_out, (h_n, h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

# torch.manual_seed(1)
# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
#
# # initialize the hidden state.
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)

if __name__ == '__main__':
    inputs = torch.randn(32,512,6)
    model = RNN(num_classes=2)
    model.eval()
    output = model(inputs)
    print(output)
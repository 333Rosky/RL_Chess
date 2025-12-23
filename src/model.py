import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=4, num_channels=64, input_channels=19, num_actions=4672):
        """
        AlphaZero style network.
        input_channels: 
            6 own pieces
            6 opp pieces
            2 repetitions
            4 castling rights
            1 side to move
            = 19
        """
        super(ChessNet, self).__init__()
        
        # Initial block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual towers
        self.res_blocks = nn.ModuleList([ResNetBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1, stride=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, num_actions)
        
        # Value Head
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1, stride=1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 64)
        self.fc_value2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (N, C, 8, 8)
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)
        
        for block in self.res_blocks:
            out = block(out)
            
        # Policy head
        policy = self.conv_policy(out)
        policy = self.bn_policy(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        # We return logits for policy, softmax is applied later usually, but log_softmax is good for NLLLoss
        
        # Value head
        value = self.conv_value(out)
        value = self.bn_value(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = self.fc_value1(value)
        value = F.relu(value)
        value = self.fc_value2(value)
        value = torch.tanh(value) # Value is between -1 and 1
        
        return policy, value

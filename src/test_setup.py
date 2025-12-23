
import sys
import os
import torch
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

from src.game import Game
from src.model import ChessNet
from src.mcts import MCTS
from src.utils import get_num_actions

def test_integration():
    print("Initializing components...")
    game = Game()
    num_actions = get_num_actions()
    print(f"Number of actions: {num_actions}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = ChessNet(num_actions=num_actions).to(device)
    mcts = MCTS(model, num_simulations=10, device=device)
    
    print("Running MCTS search...")
    probs = mcts.search(game)
    
    best_move_idx = np.argmax(probs)
    print(f"Best move index: {best_move_idx}")
    print("Probabilities sum:", np.sum(probs))
    
    print("Integration test passed!")

if __name__ == "__main__":
    test_integration()

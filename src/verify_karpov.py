import torch
import chess
import numpy as np
from src.model import ChessNet
from src.game import Game
from src.utils import get_num_actions, decode_move

def verify():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet(num_actions=get_num_actions()).to(device)
    
    path = 'checkpoints/karpov_v1.pth'
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {path}")
    except FileNotFoundError:
        print(f"{path} not found yet.")
        return

    model.eval()
    
    # Test 1: Start Position
    game = Game()
    state = game.get_state()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = model(state_tensor)
        
    policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
    
    top_5 = np.argsort(policy)[-5:][::-1]
    print("\nTop 5 moves for Start Position:")
    for idx in top_5:
        move = decode_move(idx)
        prob = policy[idx]
        print(f"{move}: {prob:.4f}")

    # Test 2: Taxing tactics (or just check if it generates legal moves)
    # Karpov was d4 player? 
    # Let's just check if e4/d4 are top.

if __name__ == "__main__":
    verify()

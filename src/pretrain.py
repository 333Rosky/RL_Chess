import chess.pgn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

from src.model import ChessNet
from src.game import Game
from src.utils import encode_move, get_num_actions

# Configuration
PGN_FILE = 'Karpov.pgn'
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
CHECKPOINT_PATH = 'checkpoints/karpov_v1.pth'
MAX_GAMES = 2000 # Limit to avoid OOM if file is massive

class PGNDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, v = self.examples[idx]
        return torch.tensor(state, dtype=torch.float32), \
               torch.tensor(pi, dtype=torch.float32), \
               torch.tensor(v, dtype=torch.float32)

def parse_pgn_data(pgn_file, max_games=None):
    print(f"Parsing {pgn_file}...")
    examples = []
    games_processed = 0
    
    with open(pgn_file) as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception as e:
                print(f"Error reading game: {e}")
                break
                
            if game is None:
                break
                
            games_processed += 1
            if max_games and games_processed > max_games:
                break
            
            if games_processed % 100 == 0:
                print(f"Processed {games_processed} games...")
                
            # Get Result
            result = game.headers.get("Result")
            if result == "1-0":
                game_value = 1
            elif result == "0-1":
                game_value = -1
            elif result == "1/2-1/2":
                game_value = 0
            else:
                continue # Unknown result
            
            # Replay game
            board = game.board()
            
            for move in game.mainline_moves():
                # Encoding State
                # We reuse Game class just for get_state convenience
                g = Game()
                g.board = board
                
                state = g.get_state()
                
                # Encoding Policy
                move_idx = encode_move(move)
                if move_idx is None:
                    board.push(move)
                    continue
                    
                pi = np.zeros(get_num_actions(), dtype=np.float32)
                pi[move_idx] = 1.0
                
                # Value Target
                # If current player (board.turn) wins match, target=+1. Else -1.
                current_val = game_value if board.turn == chess.WHITE else -game_value
                
                examples.append((state, pi, current_val))
                
                board.push(move)
                
    print(f"Finished. Processed {games_processed} games. Generated {len(examples)} examples.")
    return examples

def train():
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Data
    examples = parse_pgn_data(PGN_FILE, MAX_GAMES)
    if not examples:
        print("No examples found. Check PGN file.")
        return

    dataset = PGNDataset(examples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    num_actions = get_num_actions()
    model = ChessNet(num_actions=num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
            
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for states, pis, vs in loop:
            states, pis, vs = states.to(device), pis.to(device), vs.to(device)
            
            out_pi, out_v = model(states)
            
            # Loss
            l_pi = -torch.sum(pis * F.log_softmax(out_pi, dim=1)) / pis.size(0)
            l_v = F.mse_loss(out_v.view(-1), vs)
            
            loss = l_pi + l_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()

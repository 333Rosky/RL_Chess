import chess
import chess.engine
import torch
import numpy as np
import os
import argparse
from src.model import ChessNet
from src.game import Game
from src.utils import get_num_actions, decode_move
from src.mcts import MCTS

# Config
STOCKFISH_PATH = r"C:\Users\romai\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe"
# Ensure it is executable: chmod +x ./stockfish/stockfish-ubuntu-x86-64-avx2

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Model
    model = ChessNet(num_actions=get_num_actions()).to(device)
    path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    # Stockfish Levels to test
    elos = [600, 1000, 1400]
    
    results = {}
    
    for elo in elos:
        print(f"\n--------------------------------")
        print(f"Testing against Stockfish Elo {elo}")
        print(f"--------------------------------")
        
        score = 0
        games_to_play = 2 # 1 White, 1 Black
        
        for i in range(games_to_play):
            # i=0: Bot White. i=1: Bot Black.
            bot_color = chess.WHITE if i == 0 else chess.BLACK
            
            game = Game()
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            
            # Configure Stockfish Elo
            # Limit strength
            engine.configure({"Skill Level": 20}) # Max skill, then limit with limit?
            # Actually UCI_LimitStrength and UCI_Elo are better
            try:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            except:
                print("Could not set Elo, using default strength (very hard!)")
            
            print(f"Game {i+1}: Bot is {'White' if bot_color == chess.WHITE else 'Black'}")
            
            while not game.board.is_game_over():
                if game.board.turn == bot_color:
                    # Bot Move
                    mcts = MCTS(model, num_simulations=100, device=device) # Fast eval
                    root_probs = mcts.search(game)
                    best_idx = np.argmax(root_probs)
                    move = decode_move(best_idx)
                    game.step(move)
                else:
                    # Stockfish Move
                    result = engine.play(game.board, chess.engine.Limit(time=0.1))
                    game.step(result.move)
                    
            engine.quit()
            
            outcome = game.board.result()
            print(f"Result: {outcome}")
            
            # Update Score
            if outcome == "1-0":
                if bot_color == chess.WHITE: score += 1
            elif outcome == "0-1":
                if bot_color == chess.BLACK: score += 1
            elif outcome == "1/2-1/2":
                score += 0.5
                
        print(f"Score vs Elo {elo}: {score}/{games_to_play}")
        results[elo] = score
        
    print("\nSummary:")
    for elo, score in results.items():
        print(f"Elo {elo}: {score}/{games_to_play}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='karpov_v1.pth')
    args = parser.parse_args()
    evaluate(args)

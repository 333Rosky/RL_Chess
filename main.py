import argparse
import os
import torch
from src.model import ChessNet
from src.trainer import Trainer
from src.utils import get_num_actions

def parse_args():
    parser = argparse.ArgumentParser(description='RL Chess Bot')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'], help='Mode: train or play')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of self-play episodes per iteration')
    parser.add_argument('--num_simulations', type=int, default=50, help='Number of MCTS simulations per move')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoints dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    # Init Model
    num_actions = get_num_actions()
    model = ChessNet(num_actions=num_actions).to(device)
    
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    else:
        # Auto-load logic
        karpov_path = os.path.join(args.checkpoint_dir, 'karpov_v1.pth')
        if os.path.exists(karpov_path):
            print(f"Auto-loading Karpov Brain: {karpov_path}")
            model.load_state_dict(torch.load(karpov_path, map_location=device))
        else:
            # Try to load latest checkpoint_{i}.pth
            pass # Or implement scan logic logic later if needed

        
    if args.mode == 'train':
        trainer_args = vars(args)
        trainer_args['device'] = device
        trainer = Trainer(model, trainer_args)
        trainer.learn()
        
    elif args.mode == 'play':
        from src.game import Game
        from src.visualizer import Visualizer
        import time
        import numpy as np
        from src.utils import encode_move, decode_move
        from src.mcts import MCTS

        # Ensure visualizer output directory exists
        if not os.path.exists('parts'):
            os.makedirs('parts') # Using 'parts' as requested by user or just standard export? 
                                 # Let's use 'visuals' as planned
        if not os.path.exists('visuals'):
            os.makedirs('visuals')

        viz = Visualizer()
        game = Game()
        
        # Human vs AI or AI vs AI?
        # Let's do AI vs AI for now to demonstrate, or ask user?
        # Creating a simple loop where we can see the board.
        
        step = 0
        print("Starting Game...")
        print(game.board)
        
        # Save initial board
        viz.add_state(game.board) # Add initial state
        viz.save_board_svg(game.board, f"visuals/step_{step}.svg")
        
        while not game.board.is_game_over():
            step += 1
            print(f"\nMove {step}")
            
            # AI Move
            mcts = MCTS(model, num_simulations=args.num_simulations, device=device)
            root_probs = mcts.search(game)
            
            # Pick best move (deterministic for play)
            best_idx = np.argmax(root_probs)
            move = decode_move(best_idx)
            
            print(f"AI chooses: {move}")
            game.step(move)
            print(game.board)
            
            # Save visual
            viz.add_state(game.board)
            # We can still save individual SVGs if we want, or stop to save space.
            # viz.save_board_svg(game.board, f"visuals/step_{step}.svg")
            # viz.save_board_svg(game.board, "visuals/latest.svg")
            
        print("Game Over")
        print(f"Result: {game.board.result()}")
        
        # Save Replay
        print("Generating Replay HTML...")
        viz.save_game_html("visuals/replay.html")
        print(f"Replay saved to {os.path.abspath('visuals/replay.html')}")

if __name__ == "__main__":
    main()

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
        
    if args.mode == 'train':
        trainer_args = vars(args)
        trainer_args['device'] = device
        trainer = Trainer(model, trainer_args)
        trainer.learn()
        
    elif args.mode == 'play':
        # Simple play mode against human or random?
        pass

if __name__ == "__main__":
    main()

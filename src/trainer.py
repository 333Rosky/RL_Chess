import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from src.game import Game
from src.mcts import MCTS
from src.utils import encode_move

class ChessDataset(Dataset):
    def __init__(self, examples):
        """
        examples: list of (state, pi, v)
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, v = self.examples[idx]
        return torch.tensor(state, dtype=torch.float32), \
               torch.tensor(pi, dtype=torch.float32), \
               torch.tensor(v, dtype=torch.float32)

class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(self.model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        self.device = args['device']
        self.model.to(self.device)

    def execute_episode(self):
        """
        Runs one episode of self-play.
        Returns: list of (state, pi, v) for the episode.
        """
        game_examples = []
        game = Game()
        mcts = MCTS(self.model, num_simulations=self.args['num_simulations'], device=self.device)
        
        step = 0
        while True:
            step += 1
            # Temperature control
            temp = 1.0 if step < 30 else 0.1 # Exploit more later in game
            
            # Run MCTS
            # Note: we should probably reuse MCTS tree if careful, but rebuilding is safer for simplicity
            mcts = MCTS(self.model, num_simulations=self.args['num_simulations'], device=self.device)
            root_probs = mcts.search(game) # This returns search probs
            
            # Apply temperature
            # If temp is small, argmax. If temp is 1, sample.
            # Simplified: Use probs directly if temp ~ 1. 
            # If temp -> 0, set best move prob to 1.
            
            if temp == 0:
                best_idx = np.argmax(root_probs)
                pi = np.zeros_like(root_probs)
                pi[best_idx] = 1
            else:
                # Apply temp to probabilities
                # pi = probs ^ (1/temp) / sum(...)
                # But typically we just sample for self-play?
                # AlphaZero: first 30 moves proportional to count, then max.
                if step < 30:
                     pi = root_probs
                else:
                     best_idx = np.argmax(root_probs)
                     pi = np.zeros_like(root_probs)
                     pi[best_idx] = 1

            # Store example (state, pi, value_placeholder)
            # Value will be filled when game ends
            game_examples.append([game.get_state(), pi, None])
            
            # Choose action
            # If we are training, we sample? Or pick max?
            # AlphaZero samples for exploration during early self-play
            if step < 30:
                action_idx = np.random.choice(len(pi), p=pi)
            else:
                action_idx = np.argmax(pi)
            
            move = decode_move(action_idx)
            
            # Step
            _, reward, done = game.step(move)
            
            if done:
                # Result is reward relative to current player?
                # get_reward() returns 1 (White wins), -1 (Black wins), 0 (Draw).
                # If White won, v = 1 for White states, -1 for Black states.
                return_examples = []
                for i, (state, pi_p, _) in enumerate(game_examples):
                    # Who moved in this state?
                    # Start state (i=0) is White. i=1 is Black.
                    # color: 0=White, 1=Black.
                    # If i is even, it was White's turn. 
                    # If White won (reward=1), then v=1.
                    # If Black won (reward=-1), then v=-1.
                    # So v = reward * (1 if white else -1) -> reward.
                    
                    # If i is odd, it was Black's turn.
                    # If White won (reward=1), it is BAD for Black. v=-1.
                    # If Black won (reward=-1), it is GOOD for Black. v=1.
                    # So v = reward * (-1 if white else 1) -> -reward.
                    
                    # General: v = reward * (-1)^i
                    # Wait.
                    # i=0 (White): reward=1 -> v=1. reward=-1 -> v=-1. Correct.
                    # i=1 (Black): reward=1 -> v=-1. reward=-1 -> v=1. Correct.
                    
                    v = reward * ((-1) ** i) 
                    return_examples.append((state, pi_p, v))
                    
                return return_examples

    def train(self, examples):
        """
        Train the model on examples.
        examples: list of (state, pi, v)
        """
        dataset = ChessDataset(examples)
        loader = DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=True)
        
        self.model.train()
        total_loss = 0
        
        for batch_idx, (states, pis, vs) in enumerate(loader):
            states, pis, vs = states.to(self.device), pis.to(self.device), vs.to(self.device)
            
            out_pi, out_v = self.model(states)
            
            # Loss: 
            # Policy: Cross entropy (or NLL if log_softmax)
            # Value: MSE
            
            # out_pi is raw logits? In model.py:
            # policy = self.fc_policy(policy) -> Logits
            # value = torch.tanh(value) -> (-1, 1)
            
            l_pi = -torch.sum(pis * F.log_softmax(out_pi, dim=1)) / pis.size(0)
            l_v = F.mse_loss(out_v.view(-1), vs)
            
            loss = l_pi + l_v
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def learn(self):
        """
        Main learning loop
        """
        for i in range(self.args['num_iterations']):
            print(f"Starting iteration {i+1}...")
            iteration_examples = []
            
            # Self-play
            for e in range(self.args['num_episodes']):
                print(f"  Self-play episode {e+1}/{self.args['num_episodes']}")
                examples = self.execute_episode()
                iteration_examples.extend(examples)
            
            # Train
            print(f"  Training on {len(iteration_examples)} examples...")
            loss = self.train(iteration_examples)
            print(f"  Iteration {i+1} Loss: {loss:.4f}")
            
            # Save checkpoint
            torch.save(self.model.state_dict(), os.path.join(self.args['checkpoint_dir'], f"checkpoint_{i}.pth"))

from src.utils import decode_move

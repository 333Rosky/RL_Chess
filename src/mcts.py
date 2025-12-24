import math
import chess
import numpy as np
import torch
from src.utils import encode_move, decode_move

class Node:
    def __init__(self, parent=None, prob=0.0):
        self.parent = parent
        self.children = {}  # move_idx -> Node
        self.visit_count = 0
        self.value_sum = 0
        self.prob = prob  # Prior probability from NN

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0, device='cuda'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, game):
        root = Node()
        
        # Expand root first
        self._expand(root, game)

        for _ in range(self.num_simulations):
            node = root
            scratch_game = game.clone()
            
            # Selection
            while not node.is_leaf():
                action_idx, node = self._select_child(node)
                # Apply move
                move = decode_move(action_idx)
                scratch_game.step(move)

            # Expansion and Evaluation
            value = self._expand(node, scratch_game)

            # Backpropagation
            self._backpropagate(node, value)

        # Select best move
        # Usually we return probabilities proportional to visit counts (for training)
        # or max visit count (for play)
        
        counts = {action: child.visit_count for action, child in root.children.items()}
        total = sum(counts.values())
        probs = np.zeros(self.model.fc_policy.out_features) # Assuming model has this attribute or we get it from utils
        
        for action, count in counts.items():
            probs[action] = count / total
            
        return probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.c_puct * child.prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            # Value is from perspective of player who moved to get to this node?
            # AlphaZero: Q(s,a) + U(s,a).
            # child.value() is value of state after taking action.
            # If current player is White, child node is Black's turn. 
            # child.value() returns expected outcome for BLACK.
            # So Q should be -child.value() (because it's good for White if bad for Black).
            
            q = -child.value()
            
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _expand(self, node, game):
        # Check if game over
        _, reward, done = game.get_state(), game.get_reward(), game.board.is_game_over()
        if done:
            # If game is over, we cannot expand. The value is the reward.
            # Reward is +1 if White won, -1 if Black won.
            # If it's White's turn in 'game' (which means we are essentially at a terminal state looking for moves),
            # Wait, if game is over after the move was made to get here.
            # 'game' state corresponds to 'node'.
            # If 'node' is terminal, we return value relative to the player whose turn it WOULD be.
            # Or simpler:
            # If White won, value is +1.
            # If it is White's turn, value is +1.
            # If it is Black's turn, value is -1 (from Black's perspective).
            
            turn = game.board.turn
            if reward == 1: # White won
                return 1 if turn == chess.WHITE else -1
            elif reward == -1: # Black won
                return 1 if turn == chess.BLACK else -1
            else:
                return 0

        # Run NN
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(state_tensor)
            
        # Policy is logits, softmax it
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        value = value.item()

        # Mask invalid moves
        valid_moves = game.get_valid_moves()
        valid_indices = []
        for move in valid_moves:
            # encode_move might return None if move logic is imperfect in utils
            idx = encode_move(move)
            if idx is not None:
                valid_indices.append(idx)
        
        policy_sum = 0
        for idx in valid_indices:
            # Create child nodes
            if idx not in node.children:
                node.children[idx] = Node(parent=node, prob=policy[idx])
                policy_sum += policy[idx]
        
        # Renormalize probs if needed (optional but good)
        if policy_sum > 0:
            for child in node.children.values():
                child.prob /= policy_sum
                
        return value

    def _backpropagate(self, node, value):
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            curr = curr.parent
            value = -value # Toggle perspective

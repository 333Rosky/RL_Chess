import chess
import numpy as np

# Direction mappings (simplified for demonstration, or we can use full 4672)
# But implementing full 4672 is complex.
# Let's try to implement a FULL robust mapping because the user wants "Best possible".

def create_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1+t, n1+t) for t in range(-7, 8)] + \
                           [(l1+t, n1-t) for t in range(-7, 8)] + \
                           [(l1+t, n1+2*t) for t in [-1, 1]] + \
                           [(l1+2*t, n1+t) for t in [-1, 1]] + \
                           [(l1+t, n1-2*t) for t in [-1, 1]] + \
                           [(l1+2*t, n1-t) for t in [-1, 1]] # This covers knights too broadly but okay

    # Actually, simpler approach: Generate ALL possible moves in chess (from any square to any square + promotions)
    # A move is (from_square, to_square, promotion).
    # Total distinct moves roughly ~1800-2000? 
    # But for a fixed output vector, we usually iterate all squares.
    
    # Let's just generate a flat list of all unique UCI strings that *could* happen.
    # From sq (64) * To sq (63?) + promotions.
    
    labels = []
    for f_rank in range(1, 9):
        for f_file in letters:
            for t_rank in range(1, 9):
                for t_file in letters:
                    f_sq = f"{f_file}{f_rank}"
                    t_sq = f"{t_file}{t_rank}"
                    if f_sq == t_sq: continue
                    
                    # Normal move
                    labels.append(f"{f_sq}{t_sq}")
                    
                    # Promotions (only if from rank 7->8 or 2->1? No, we need to list ALL theoretical promotions)
                    # Actually standard alphazero output is relative to input square.
                    # Implementing that perfectly is tedious.
                    # Alternate approach: Just map ALL 1968 possible UCI moves to indices?
                    # Max legal moves in a position is ~218.
                    # Total unique moves in chess is 1972 without promotions? No.
                    
                    # Let's include promotions for relevant squares
                    if (f_rank == 7 and t_rank == 8) or (f_rank == 2 and t_rank == 1):
                         for p in promoted_to:
                             labels.append(f"{f_sq}{t_sq}{p}")
                             
    return labels

# Let's optimize: We only need to map the legal moves of the current position to indices.
# But the NN outputs a fixed size vector.
# Let's use the 4672 size.
# Actually, for this implementation, let's use a simpler discrete action space if allowed?
# The user wants "best possible". AlphaZero uses 4672.
# Let's try to implement a mapping that covers most cases or finding a library?
# No external library for this specific mapping usually.

# Let's stick to a slightly simpler index: 
# unique_moves = [ all from-to combinations ] + [ promotions ]
# 64*64 = 4096. + promotions (64*3? no, only pawns).
# ~4200. This is close to 4672.

def get_uci_labels():
    labels = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for f_let in letters:
        for f_num in numbers:
            f_sq = f_let + f_num
            for t_let in letters:
                for t_num in numbers:
                    t_sq = t_let + t_num
                    if f_sq != t_sq:
                        labels.append(f_sq + t_sq)
                        # Check for potential promotion
                        # We just add them all to be safe?
                        # If we just add 'q', 'r', 'b', 'n' to ALL moves it's 4096 * 5 ~ 20k. Too big.
                        
                        # Only specific ranks.
                        if (f_num == '7' and t_num == '8') or (f_num == '2' and t_num == '1'):
                            # Wait, we don't know if it's a pawn or not here.
                            # So we must include them in the label set corresponding to the output of the NN.
                            for p in promoted_to:
                                labels.append(f_sq + t_sq + p)
                                
    return list(dict.fromkeys(labels)) # remove duplicates if any

UCI_LABELS = get_uci_labels()
LABEL_TO_INDEX = {label: i for i, label in enumerate(UCI_LABELS)}
INDEX_TO_LABEL = {i: label for i, label in enumerate(UCI_LABELS)}

def encode_move(move):
    return LABEL_TO_INDEX.get(move.uci(), None) # Might return None if not in list (shouldn't happen if list is complete)

def decode_move(index):
    return chess.Move.from_uci(INDEX_TO_LABEL[index])

def get_num_actions():
    return len(UCI_LABELS)

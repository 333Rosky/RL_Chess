import chess
import numpy as np

class Game:
    def __init__(self):
        self.board = chess.Board()

    def get_init_board(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        """
        Returns the state of the board as a numpy array (19, 8, 8).
        Planes:
        0-5: Own pieces (P, N, B, R, Q, K)
        6-11: Opp pieces (P, N, B, R, Q, K)
        12-13: Repetitions (1, 2)
        14-17: Castling rights (K, Q, k, q) (own then opp) - wait, encoded relative to color?
        Let's stick to absolute for simplicity or relative? 
        AlphaZero uses relative features (current player always "own").
        
        Let's use relative:
        0-5: Current player pieces
        6-11: Opponent pieces
        12: Repetition count >= 1
        13: Repetition count >= 2
        14: Color (0 for white, 1 for black) - actually simpler to just own/opp
        15: Total move count (normalized)? Or just constant 1 plane?
        16-19: Castling (Own K, Own Q, Opp K, Opp Q)
        
        Actually, let's fix the 19 planes as defined in model.py docstring:
        0-5: Own pieces
        6-11: Opp pieces
        12-13: Repetitions (1, 2)
        14-17: Castling rights
        18: Side to move (all 0 if white, all 1 if black)
        """
        
        state = np.zeros((19, 8, 8), dtype=np.float32)
        
        # Helper to fill planes
        def fill_pieces(color, offset):
            for i, piece_type in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
                for square in self.board.pieces(piece_type, color):
                    row, col = divmod(square, 8)
                    state[offset + i][7 - row][col] = 1 # python-chess uses rank 0 as bottom, but matrix 0 is top. 
                                                        # Let's map rank 0 to index 7? Or standard?
                                                        # Standard FEN: rank 8 is top.
                                                        # python-chess: square 0 is A1.
                                                        # Matrix: (0,0) usually top-left.
                                                        # Let's map A1 (0) to (7, 0)
        
        turn = self.board.turn
        
        # 0-5: Own pieces
        fill_pieces(turn, 0)
        
        # 6-11: Opp pieces
        fill_pieces(not turn, 6)
        
        # 12-13: Repetitions
        # This is expensive to check fully correctly without history, but board holds some history?
        # self.board.is_repetition(1) etc.
        if self.board.is_repetition(2):
            state[12][:][:] = 1
        if self.board.is_repetition(3):
            state[13][:][:] = 1
            
        # 14-17: Castling rights (Own K, Own Q, Opp K, Opp Q)
        if self.board.has_kingside_castling_rights(turn):
            state[14][:][:] = 1
        if self.board.has_queenside_castling_rights(turn):
            state[15][:][:] = 1
        if self.board.has_kingside_castling_rights(not turn):
            state[16][:][:] = 1
        if self.board.has_queenside_castling_rights(not turn):
            state[17][:][:] = 1
            
        # 18: Side to move
        if turn == chess.BLACK:
            state[18][:][:] = 1
            
        return state

    def get_valid_moves(self):
        return list(self.board.legal_moves)
    
    def step(self, move):
        self.board.push(move)
        return self.get_state(), self.get_reward(), self.board.is_game_over()

    def get_reward(self):
        outcome = self.board.outcome()
        if outcome is None:
            return 0
        if outcome.winner == self.board.turn: # This shouldn't happen immediately after push? 
                                              # Actually after push, it's opponent's turn. 
                                              # If winner is turn (opponent), then we (previous player) lost.
            # But the value is usually view from "current player to move".
            # If game is over, and winner is NOT turn, then loop player won.
            pass
            
        # Simpler: 
        # If White won, +1. If Black won, -1.
        # But MCTS usually expects value for the current player.
        
        if outcome.winner == chess.WHITE:
            return 1
        elif outcome.winner == chess.BLACK:
            return -1
        else:
            return 0 # Draw
            
    def get_fen(self):
        return self.board.fen()
        
    def clone(self):
        g = Game()
        g.board = self.board.copy()
        return g

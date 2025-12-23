import chess
import chess.svg

class Visualizer:
    def __init__(self, export_dir='visuals'):
        self.export_dir = export_dir
        self.history = [] # List of FEN strings
        
    def add_state(self, board):
        self.history.append(board.fen())
        
    def save_board_svg(self, board, filename):
        """Save the board as an SVG file"""
        board_svg = chess.svg.board(board=board)
        with open(filename, "w") as f:
            f.write(board_svg)

    def save_game_html(self, filename):
        """Generates an HTML file with a replay viewer."""
        
        # Pre-generate SVGs for all moves could be heavy if embedded directly as strings.
        # Alternatively, we can use chess.js and chessboard.js in the HTML, 
        # but that requires internet access or local libs.
        # Safer approach for standalone: Embed SVGs as array of strings.
        
        svg_frames = []
        board = chess.Board()
        
        # Initial state
        # We stored FENs. Let's recreate boards to get SVGs.
        # Or did we store board objects? FEN is safer for cloning.
        
        for fen in self.history:
            board.set_fen(fen)
            svg = chess.svg.board(board=board)
            # Escape for JS string
            svg = svg.replace('"', "'").replace('\n', '')
            svg_frames.append(svg)
            
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chess Replay</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; background-color: #222; color: #eee; }}
        #board {{ width: 600px; height: 600px; margin: 20px auto; }}
        .controls {{ margin: 20px; }}
        button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; background: #444; color: #fff; border: 1px solid #666; }}
        button:hover {{ background: #555; }}
        #status {{ margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>Chess Replay</h1>
    <div id="board"></div>
    <div class="controls">
        <button onclick="first()">|&lt;</button>
        <button onclick="prev()">&lt;</button>
        <button onclick="playPause()" id="playBtn">Play</button>
        <button onclick="next()">&gt;</button>
        <button onclick="last()">&gt;|</button>
    </div>
    <div id="status">Move: <span id="moveNum">0</span> / <span id="totalMoves">{len(svg_frames)-1}</span></div>

    <script>
        const frames = [
            "{ '", "'.join(svg_frames) }"
        ];
        
        // Fix the join above which is python syntax inside f-string... wait.
        // I can't easily join with quotes inside the f-string like that.
        // Let's rely on Python to format the list properly or loop.
    </script>
    <script>
        // Data injection
        const raw_frames = {svg_frames}; 
        // raw_frames will be a JS array of strings because python list str() looks like ['a', 'b']
        
        let currentFrame = 0;
        let playing = false;
        let interval = null;
        
        function showFrame(index) {{
            if (index < 0) index = 0;
            if (index >= raw_frames.length) index = raw_frames.length - 1;
            currentFrame = index;
            document.getElementById('board').innerHTML = raw_frames[index];
            document.getElementById('moveNum').innerText = currentFrame;
        }}
        
        function next() {{ showFrame(currentFrame + 1); }}
        function prev() {{ showFrame(currentFrame - 1); }}
        function first() {{ showFrame(0); }}
        function last() {{ showFrame(raw_frames.length - 1); }}
        
        function playPause() {{
            if (playing) {{
                clearInterval(interval);
                document.getElementById('playBtn').innerText = "Play";
                playing = false;
            }} else {{
                interval = setInterval(() => {{
                    if (currentFrame < raw_frames.length - 1) {{
                        next();
                    }} else {{
                        playPause(); // Stop at end
                    }}
                }}, 800);
                document.getElementById('playBtn').innerText = "Pause";
                playing = true;
            }}
        }}
        
        // Init
        showFrame(0);
        
        document.addEventListener('keydown', function(e) {{
            if (e.key === "ArrowLeft") prev();
            if (e.key === "ArrowRight") next();
            if (e.key === " ") playPause();
        }});
    </script>
</body>
</html>
        """
        
        # Need to fix the JS array injection.
        # Using json.dumps is safest.
        import json
        json_frames = json.dumps(svg_frames)
        
        # Re-construct HTML with correct injection
        html_content = html_content.replace(f"const raw_frames = {svg_frames};", f"const raw_frames = {json_frames};")
        
        with open(filename, "w") as f:
            f.write(html_content)


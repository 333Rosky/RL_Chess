import os

def find_stockfish():
    # Search in common locations
    search_roots = [
        r"C:\Users\romai",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
    ]
    
    target_name = "stockfish-windows-x86-64-avx2.exe"
    target_name_simple = "stockfish.exe"

    print("Searching for Stockfish...")
    
    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            continue
            
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file == target_name or file == target_name_simple:
                    full_path = os.path.join(root, file)
                    print(f"FOUND: {full_path}")
                    with open("stockfish_path.txt", "w") as f:
                        f.write(full_path)
                    return full_path
                    
    print("NOT FOUND")

if __name__ == "__main__":
    find_stockfish()

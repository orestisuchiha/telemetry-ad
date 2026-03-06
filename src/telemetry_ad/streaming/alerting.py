from pathlib import Path


def append_alert(log_file: str, timestamp: str, score: float, threshold: float, model: str) -> None:
    path = Path(log_file)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,model,score,threshold\n")
        f.write(f"{timestamp},{model},{score:.6f},{threshold:.6f}\n")

def classify_event_type(run_length: int, min_collective: int = 3) -> str:
    if run_length <= 1:
        return "point"
    if run_length >= min_collective:
        return "collective"
    return "contextual"

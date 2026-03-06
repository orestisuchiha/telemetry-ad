from sklearn.ensemble import IsolationForest


def make_iforest(seed: int = 42) -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )

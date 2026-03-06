import pickle
from src.config import Paths, DataCfg, TrainCfg
from src.data_utils import load_taxi_csv, make_windows, chronological_split, fit_standardizer_on_train, apply_standardizer

def main():
    paths = Paths()
    data_cfg = DataCfg()

    df = load_taxi_csv(str(paths.DATA_CSV))
    X, y, ts = make_windows(df, data_cfg.WINDOW, data_cfg.STRIDE, data_cfg.USE_TIME_FEATURES)
    split = chronological_split(X, y, ts, data_cfg.TRAIN_FRAC, data_cfg.VAL_FRAC)

    scaler = fit_standardizer_on_train(split.X_train)
    split.X_train = apply_standardizer(split.X_train, scaler)
    split.X_val   = apply_standardizer(split.X_val, scaler)
    split.X_test  = apply_standardizer(split.X_test, scaler)

    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.OUT_DIR / "split.pkl", "wb") as f:
        pickle.dump({"split": split, "scaler": scaler}, f)

    print("Saved:", paths.OUT_DIR / "split.pkl")
    print("Shapes:", split.X_train.shape, split.X_val.shape, split.X_test.shape)

if __name__ == "__main__":
    main()
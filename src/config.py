# Here we put hyperparameters, paths, and experimen constants
# Hyperparameters : LofCfg, TrainCfg, DataCfg 
# Split : Chronological split(no shuffle) -> The data is temporal, 30 days for every 30 min data. We used 60/20/20 split
# Inductive bias : daily pattern of morning/evening taxi demand taken into account, we added sin/cos time features to encode cyclic preiodicity

# Hyperparameters are currently fixed. Future iterations will use Vertex AI hyperparameter tuning to automatically search the parameter space and optimize validation performance

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_CSV: Path = ROOT / "data" / "nyc_taxi.csv"
    OUT_DIR: Path = ROOT / "outputs"
    FIG_DIR: Path = OUT_DIR / "figures"
    TAB_DIR: Path = OUT_DIR / "tables"
    MODEL_DIR: Path = ROOT / "models"

@dataclass(frozen=True)
class DataCfg:
    
    WINDOW: int = 48               # 48 * 30min = 24 hours
    STRIDE: int = 1
    # chronological splits
    TRAIN_FRAC: float = 0.60
    VAL_FRAC: float = 0.20         # test = rest
    # feature options
    USE_TIME_FEATURES: bool = True # sin/cos hour + day-of-week

@dataclass(frozen=True)
class TrainCfg:
    SEED: int = 42
    DEVICE: str = "cuda"           

    # RNN
    RNN_EPOCHS: int = 5
    RNN_LR: float = 1e-3
    RNN_HIDDEN: int = 64
    RNN_LAYERS: int = 1
    RNN_BATCH: int = 256
    RNN_DROPOUT: float = 0.0

    #VAE
    VAE_EPOCHS: int = 10
    VAE_LR: float = 1e-3
    VAE_HIDDEN: int = 128
    VAE_Z: int = 16
    VAE_BETA: float = 1.0
    VAE_BATCH: int = 256

@dataclass(frozen=True)
class LofCfg:
    N_NEIGHBORS: int = 35
    CONTAMINATION: float = 0.1     # used only for internal LOF behavior
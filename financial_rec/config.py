import os
import datetime
from typing import Dict

# --- Configuration ---

# Output Configuration
OUTPUT_DIR = "financial_rec/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_data.json")
NUM_SAMPLES = 200  # Number of samples to generate

# Data Generation Parameters
SECURITIES = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "JPM", "BAC", "WFC"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD"]
BROKERS = ["BrokerA", "BrokerB", "BrokerC", "Internal"]
BASE_DATE = datetime.date(2026, 1, 1)
DATE_RANGE_DAYS = 60
MIN_TRADES_PER_POS = 1
MAX_TRADES_PER_POS = 4
MIN_QTY = 10
MAX_QTY = 500
MIN_PRICE = 5
MAX_PRICE = 1000
ACCOUNT_NO_PROB = 0.8 # Probability a trade/position has an account number
BROKER_PROB = 0.9 # Probability a trade/position has a broker assigned

# Mismatch Tolerances
PRICE_TOLERANCE_PERCENT = 0.01 # 1%
DATE_TOLERANCE_DAYS = 1

# Mismatch Proportions (ensure values sum roughly to 1.0)
# Controls the probability of *attempting* to create each mismatch type.
# The actual final distribution might vary slightly if certain mismatches can't be applied.
MISMATCH_PROPORTIONS: Dict[str, float] = {
    "NoBreak": 0.40,
    "PriceMismatch": 0.10,
    "QuantityMismatch": 0.10, # Applied only on single-trade samples for now
    "MissingTrade": 0.10,     # Applied only on multi-trade samples
    "ExtraTrade": 0.05,       # TODO: Implement logic for ExtraTrade
    "FXMismatch": 0.07,
    "DateMismatch": 0.08,
    "AccountError": 0.05,     # Novel code
    "BrokerMismatch": 0.05,   # Novel code
}

# --- Derived Configuration (Calculated from above) ---

# Normalize probabilities to ensure they sum to 1
_total_prob = sum(MISMATCH_PROPORTIONS.values())
NORMALIZED_PROPORTIONS = {k: v / _total_prob for k, v in MISMATCH_PROPORTIONS.items()}
MISMATCH_TYPES = list(NORMALIZED_PROPORTIONS.keys())
MISMATCH_WEIGHTS = list(NORMALIZED_PROPORTIONS.values())

# --- Other Potential Configuration (Add as needed) ---
# LOG_LEVEL = "INFO"
# WANDB_PROJECT = "FinancialRecRL" 
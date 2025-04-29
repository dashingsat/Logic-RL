import json
import random
import datetime
import uuid
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# Import configuration
import financial_rec.config as cfg

# --- Configuration ---

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
# The actual final distribution might vary slightly if certain mismatches can't be applied (e.g., QuantityMismatch on multi-trade).
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
# Normalize probabilities to ensure they sum to 1
total_prob = sum(MISMATCH_PROPORTIONS.values())
normalized_proportions = {k: v / total_prob for k, v in MISMATCH_PROPORTIONS.items()}
mismatch_types = list(normalized_proportions.keys())
mismatch_weights = list(normalized_proportions.values())


# --- Data Structures ---

@dataclass
class Trade:
    trade_id: str = field(default_factory=lambda: f"T{uuid.uuid4().hex[:6].upper()}")
    security: str = "AAPL"
    quantity: float = 100.0
    price: float = 150.00
    trade_date: str = "2026-01-10"
    currency: str = "USD"
    account_no: Optional[str] = None
    broker: Optional[str] = None # Added broker field

@dataclass
class Position:
    position_id: str = field(default_factory=lambda: f"P{uuid.uuid4().hex[:6].upper()}")
    security: str = "AAPL"
    total_quantity: float = 100.0
    position_price: float = 150.00 # Typically represents the weighted average price or closing price
    position_date: str = "2026-01-10"
    currency: str = "USD"
    account_no: Optional[str] = None
    broker: Optional[str] = None # Added broker field

@dataclass
class ReconciliationSample:
    trades: List[Dict[str, Any]] # Store as dicts for direct JSON compatibility
    position: Dict[str, Any]   # Store as dict
    label: str                 # Ground truth reason code
    mismatch_details: Optional[Dict[str, Any]] = None # Added field for details
    # metadata: Optional[Dict[str, Any]] = None # Optional: Store generation info

# --- Helper Functions ---

def generate_random_date(base_date, range_days):
    """Generates a random date within a range."""
    offset = random.randint(0, range_days)
    return (base_date + datetime.timedelta(days=offset)).isoformat()

def create_mismatch(
    trades: List[Trade],
    position: Position,
    mismatch_type: str
) -> tuple[List[Trade], Position, str, Optional[Dict[str, Any]]]:
    """
    Modifies *copies* of trades or position to create a specific mismatch.
    Returns the modified trades, position, and the final applied label.
    Note: Some mismatches might not be applicable (e.g., MissingTrade if only 1 trade exists),
          in which case it might return the original data and potentially a "NoBreak" label.
    """
    mod_trades = [Trade(**asdict(t)) for t in trades] # Operate on copies
    mod_pos = Position(**asdict(position))            # Operate on a copy
    applied_label = mismatch_type # Assume success initially
    details: Optional[Dict[str, Any]] = None # Initialize details

    num_trades = len(mod_trades)

    # Calculate initial aggregate values (needed for details)
    initial_trade_qty_sum = round(sum(t.quantity for t in trades), 2)
    initial_trade_value_sum = round(sum(t.quantity * t.price for t in trades), 2)
    initial_avg_trade_price = round(initial_trade_value_sum / initial_trade_qty_sum, 2) if initial_trade_qty_sum else 0
    initial_trade_date = trades[0].trade_date if trades else None
    initial_trade_currencies = list(set(t.currency for t in trades))
    initial_trade_accounts = list(set(t.account_no for t in trades if t.account_no))
    initial_trade_brokers = list(set(t.broker for t in trades if t.broker))

    # --- Apply Mismatch Logic ---
    try:
        if mismatch_type == "NoBreak":
            pass # No changes needed

        elif mismatch_type == "PriceMismatch":
            expected_pos_price = position.position_price
            delta = expected_pos_price * (cfg.PRICE_TOLERANCE_PERCENT + random.uniform(0.005, 0.05))
            mod_pos.position_price = round(expected_pos_price + delta * random.choice([-1, 1]), 2)
            mod_pos.position_price = max(0.01, mod_pos.position_price)
            details = {
                "component": "price",
                "trade_value_avg": initial_avg_trade_price,
                "position_value": mod_pos.position_price,
                "tolerance_percent": cfg.PRICE_TOLERANCE_PERCENT
            }

        elif mismatch_type == "QuantityMismatch":
            if num_trades == 1:
                trade_qty = mod_trades[0].quantity
                delta = max(1, int(trade_qty * random.uniform(0.05, 0.2)))
                mod_pos.total_quantity = round(trade_qty + delta * random.choice([-1, 1]), 2)
                mod_pos.total_quantity = max(0, mod_pos.total_quantity)
                details = {
                    "component": "quantity",
                    "trade_value_sum": trade_qty,
                    "position_value": mod_pos.total_quantity
                }
            else:
                applied_label = "NoBreak"
                details = None

        elif mismatch_type == "MissingTrade":
            if num_trades > 1:
                removed_trade_idx = random.randrange(num_trades)
                removed_trade = mod_trades.pop(removed_trade_idx)
                details = {
                    "component": "trade_count",
                    "expected_security": position.security,
                    "expected_date": position.position_date,
                    "removed_trade_id": removed_trade.trade_id,
                    "expected_count": num_trades,
                    "actual_count": num_trades - 1
                }
            else:
                applied_label = "NoBreak"
                details = None

        elif mismatch_type == "FXMismatch":
            original_currency = mod_pos.currency
            possible_currencies = [c for c in cfg.CURRENCIES if c != original_currency]
            changed_entity = "none"
            changed_idx = -1
            new_curr = None
            if possible_currencies:
                new_curr = random.choice(possible_currencies)
                if random.random() < 0.5 and num_trades > 0:
                    changed_idx = random.randrange(num_trades)
                    mod_trades[changed_idx].currency = new_curr
                    changed_entity = "trade"
                else:
                    mod_pos.currency = new_curr
                    changed_entity = "position"

                details = {
                    "component": "currency",
                    "trade_values": initial_trade_currencies,
                    "position_value": mod_pos.currency if changed_entity == "position" else original_currency,
                    "changed_entity": changed_entity,
                    "changed_index": changed_idx if changed_entity == "trade" else None,
                    "new_currency": new_curr
                }
            else:
                applied_label = "NoBreak"
                details = None

        elif mismatch_type == "DateMismatch":
            original_pos_date = mod_pos.position_date
            pos_dt = datetime.date.fromisoformat(original_pos_date)
            offset = random.randint(cfg.DATE_TOLERANCE_DAYS + 1, cfg.DATE_TOLERANCE_DAYS + 5)
            mod_pos.position_date = (pos_dt + datetime.timedelta(days=offset * random.choice([-1, 1]))).isoformat()
            details = {
                "component": "date",
                "trade_value": initial_trade_date,
                "position_value": mod_pos.position_date,
                "tolerance_days": cfg.DATE_TOLERANCE_DAYS
            }

        elif mismatch_type == "AccountError":
            acc1 = f"ACC{random.randint(100, 999)}"
            acc2 = f"ACC{random.randint(1000, 9999)}"
            while acc1 == acc2:
                 acc2 = f"ACC{random.randint(1000, 9999)}"
            mod_pos.account_no = acc1
            changed_trade_idx = -1
            if num_trades > 0:
                idx_to_change = random.randrange(num_trades)
                for i in range(num_trades):
                    mod_trades[i].account_no = acc1 if i != idx_to_change else acc2
                changed_trade_idx = idx_to_change
                details = {
                    "component": "account_no",
                    "trade_values": list(set(t.account_no for t in mod_trades if t.account_no)),
                    "position_value": mod_pos.account_no,
                    "differing_trade_index": changed_trade_idx
                }
            else:
                 applied_label = "NoBreak"
                 details = None

        elif mismatch_type == "BrokerMismatch":
            broker1 = random.choice(cfg.BROKERS)
            possible_brokers = [b for b in cfg.BROKERS if b != broker1]
            changed_trade_idx = -1
            if possible_brokers:
                broker2 = random.choice(possible_brokers)
                mod_pos.broker = broker1
                if num_trades > 0:
                    idx_to_change = random.randrange(num_trades)
                    for i in range(num_trades):
                        mod_trades[i].broker = broker1 if i != idx_to_change else broker2
                    changed_trade_idx = idx_to_change
                    details = {
                        "component": "broker",
                        "trade_values": list(set(t.broker for t in mod_trades if t.broker)),
                        "position_value": mod_pos.broker,
                        "differing_trade_index": changed_trade_idx
                    }
                else:
                    applied_label = "NoBreak"
                    details = None
            else:
                 applied_label = "NoBreak"
                 details = None

    except Exception as e:
        print(f"Warning: Error applying mismatch type '{mismatch_type}': {e}. Defaulting to NoBreak.")
        mod_trades = [Trade(**asdict(t)) for t in trades]
        mod_pos = Position(**asdict(position))
        applied_label = "NoBreak"
        details = {"error": str(e), "component": "generation_error"}

    # Final verification (optional but recommended)
    # TODO: Add checks here to verify if the applied modifications *actually* result
    # in the intended mismatch according to tolerances. If not, revert label to "NoBreak".
    # Example: for PriceMismatch, recalculate weighted avg price of mod_trades and check vs mod_pos.position_price

    return mod_trades, mod_pos, applied_label, details


# --- Main Generation Logic ---

def generate_data(num_samples: int) -> List[ReconciliationSample]:
    """Generates synthetic reconciliation samples using config."""
    samples = []

    print(f"Attempting to generate {num_samples} samples with target proportions:")
    for type, prop in cfg.NORMALIZED_PROPORTIONS.items():
        print(f"  - {type}: {prop:.2%}")

    # Use mismatch types and weights from config
    mismatch_types = cfg.MISMATCH_TYPES
    mismatch_weights = cfg.MISMATCH_WEIGHTS
    generated_counts = {m_type: 0 for m_type in mismatch_types}
    actual_generated_counts = {m_type: 0 for m_type in mismatch_types}

    while len(samples) < num_samples:
        # 1. Base Sample Generation (Matching)
        num_trades = random.randint(cfg.MIN_TRADES_PER_POS, cfg.MAX_TRADES_PER_POS)
        base_security = random.choice(cfg.SECURITIES)
        base_currency = random.choice(cfg.CURRENCIES)
        base_date = generate_random_date(cfg.BASE_DATE, cfg.DATE_RANGE_DAYS)
        base_account = f"ACC{random.randint(100, 999)}" if random.random() < cfg.ACCOUNT_NO_PROB else None
        base_broker = random.choice(cfg.BROKERS) if random.random() < cfg.BROKER_PROB else None

        trades: List[Trade] = []
        total_quantity = 0.0
        value_sum = 0.0

        for _ in range(num_trades):
            quantity = round(random.uniform(cfg.MIN_QTY, cfg.MAX_QTY), 2)
            price = round(random.uniform(cfg.MIN_PRICE, cfg.MAX_PRICE), 2)
            trade_date = base_date
            trade = Trade(
                security=base_security,
                quantity=quantity,
                price=price,
                trade_date=trade_date,
                currency=base_currency,
                account_no=base_account,
                broker=base_broker
            )
            trades.append(trade)
            total_quantity += quantity
            value_sum += quantity * price

        base_total_quantity = round(total_quantity, 2)
        base_position_price = round(value_sum / total_quantity, 2) if total_quantity else 0

        position = Position(
            security=base_security,
            total_quantity=base_total_quantity,
            position_price=base_position_price,
            position_date=base_date,
            currency=base_currency,
            account_no=base_account,
            broker=base_broker
        )

        # 2. Select and Apply Discrepancy
        chosen_mismatch_type = random.choices(mismatch_types, weights=mismatch_weights, k=1)[0]

        final_trades, final_position, final_label, mismatch_details = create_mismatch(
            trades, position, chosen_mismatch_type
        )

        # 3. Create Sample
        sample = ReconciliationSample(
            trades=[asdict(t) for t in final_trades],
            position=asdict(final_position),
            label=final_label,
            mismatch_details=mismatch_details
        )
        samples.append(sample)
        generated_counts[final_label] += 1
        actual_generated_counts[final_label] += 1

        if len(samples) % (num_samples // 10) == 0 and num_samples >= 10:
             print(f"Generated {len(samples)}/{num_samples} samples...")

    print("\nFinal generated counts by label:")
    for label, count in generated_counts.items():
        print(f"  - {label}: {count} ({count/num_samples:.2%})")

    print("\n--- Generation Summary ---")
    print(f"Total samples generated: {len(samples)}")
    print("Target distribution vs Actual distribution:")
    target_proportions = {k: v for k, v in zip(mismatch_types, mismatch_weights)}
    for m_type in mismatch_types:
        target_perc = target_proportions.get(m_type, 0) * 100
        actual_perc = (actual_generated_counts[m_type] / num_samples) * 100 if num_samples > 0 else 0
        print(f"  - {m_type:<20}: Target {target_perc:>5.1f}% ({generated_counts[m_type]:>4} attempts) -> Actual {actual_perc:>5.1f}% ({actual_generated_counts[m_type]:>4} generated)")
    print("--------------------------\n")

    return samples

# --- Execution ---

if __name__ == "__main__":
    print(f"--- Synthetic Data Generation ---")
    # Use config for output dir and number of samples
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    generated_data = generate_data(cfg.NUM_SAMPLES)

    output_payload = [s.__dict__ for s in generated_data]

    try:
        with open(cfg.OUTPUT_FILE, 'w') as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSuccessfully generated {len(generated_data)} samples and saved to {cfg.OUTPUT_FILE}")
    except IOError as e:
        print(f"\nError writing to file {cfg.OUTPUT_FILE}: {e}")
    except TypeError as e:
        print(f"\nError serializing data to JSON: {e}") 
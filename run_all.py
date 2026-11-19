"""Reproduce every table and figure in the paper. Usage: python run_all.py
Add --robustness for the multi-seed, bootstrap and sensitivity analyses (about one hour on 2 cores)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import generate, model, external, dynamics, robustness

if __name__ == "__main__":
    generate.main()
    model.main()
    if (Path(__file__).parent / "data" / "credit_data.csv").exists():
        external.main()
    else:
        print("data/credit_data.csv missing: run scripts/get_credit_data.sh for Section 7.6")
    dynamics.main()
    if "--robustness" in sys.argv:
        robustness.main()
        if (Path(__file__).parent / "data" / "uci_credit_card.csv").exists():
            from cli import card
            card.main()
        else:
            print("data/uci_credit_card.csv missing: run scripts/get_card_data.sh for the card-account replication")

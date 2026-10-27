"""Reproduce every table and figure in the paper. Usage: python run_all.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import generate, model, external, dynamics

if __name__ == "__main__":
    generate.main()
    model.main()
    if (Path(__file__).parent / "data" / "credit_data.csv").exists():
        external.main()
    else:
        print("data/credit_data.csv missing: run scripts/get_credit_data.sh for Section 7.6")
    dynamics.main()

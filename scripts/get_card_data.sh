#!/usr/bin/env bash
# Downloads the UCI "Default of Credit Card Clients" data set (Yeh 2009) and writes data/uci_credit_card.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL -o data/uci_card.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
python - <<'PY'
import zipfile, pandas as pd
z = zipfile.ZipFile("data/uci_card.zip")
name = [n for n in z.namelist() if n.endswith(".xls")][0]
df = pd.read_excel(z.open(name), header=1)
df.to_csv("data/uci_credit_card.csv", index=False)
print("saved data/uci_credit_card.csv", df.shape)
PY
rm -f data/uci_card.zip

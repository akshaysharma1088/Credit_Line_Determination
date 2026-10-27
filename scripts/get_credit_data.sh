#!/usr/bin/env bash
# Downloads the public credit_data benchmark (Rdatasets, modeldata package) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL -o data/credit_data.csv \
  https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/credit_data.csv
echo "saved data/credit_data.csv"

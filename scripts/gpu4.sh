#! /bin/bash

echo "===== convnext_exponent ====="; date
python scripts/expt_tuesday.py --target convnext_exponent
echo "===== convnext_exponent --adam ====="; date
python scripts/expt_tuesday.py --target convnext_exponent --adam
echo "===== convnext_exponent --poison ====="; date
python scripts/expt_tuesday.py --target convnext_exponent --poison
echo "===== convnext_exponent --poison --adam ====="; date
python scripts/expt_tuesday.py --target convnext_exponent --poison --adam

echo "===== pythia_exponent ====="; date
python scripts/expt_tuesday.py --target pythia_exponent
echo "===== pythia_exponent --adam ====="; date
python scripts/expt_tuesday.py --target pythia_exponent --adam
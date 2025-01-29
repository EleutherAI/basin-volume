#! /bin/bash

echo "===== convnext_cutoff ====="; date
python scripts/expt_tuesday.py --target convnext_cutoff
echo "===== convnext_cutoff --adam ====="; date
python scripts/expt_tuesday.py --target convnext_cutoff --adam
echo "===== convnext_cutoff --poison ====="; date
python scripts/expt_tuesday.py --target convnext_cutoff --poison
echo "===== convnext_cutoff --poison --adam ====="; date
python scripts/expt_tuesday.py --target convnext_cutoff --poison --adam

echo "===== pythia_cutoff ====="; date
python scripts/expt_tuesday.py --target pythia_cutoff
echo "===== pythia_cutoff --adam ====="; date
python scripts/expt_tuesday.py --target pythia_cutoff --adam
echo "===== pythia_cutoff --adam --eps 1e-2 ====="; date
python scripts/expt_tuesday.py --target pythia_cutoff --adam --eps 1e-2
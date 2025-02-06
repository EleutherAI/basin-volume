#! /bin/bash

echo "===== convnext_cutoff ====="; date
python scripts/expt_feb.py --target convnext_cutoff
echo "===== convnext_cutoff --adam ====="; date
python scripts/expt_feb.py --target convnext_cutoff --adam
echo "===== convnext_cutoff --poison ====="; date
python scripts/expt_feb.py --target convnext_cutoff --poison
echo "===== convnext_cutoff --poison --adam ====="; date
python scripts/expt_feb.py --target convnext_cutoff --poison --adam
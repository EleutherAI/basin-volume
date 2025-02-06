#! /bin/bash

echo "===== convnext_histo ====="; date
python scripts/expt_0205.py --target convnext_histo
echo "===== convnext_histo --adam ====="; date
python scripts/expt_0205.py --target convnext_histo --adam
echo "===== convnext_histo --poison ====="; date
python scripts/expt_0205.py --target convnext_histo --poison
echo "===== convnext_histo --poison --adam ====="; date
python scripts/expt_0205.py --target convnext_histo --poison --adam

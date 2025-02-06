#! /bin/bash

echo "===== convnext_chkpts ====="; date
python scripts/expt_feb.py --target convnext_chkpts
echo "===== convnext_chkpts --adam ====="; date
python scripts/expt_feb.py --target convnext_chkpts --adam
echo "===== convnext_chkpts --poison ====="; date
python scripts/expt_feb.py --target convnext_chkpts --poison
echo "===== convnext_chkpts --poison --adam ====="; date
python scripts/expt_feb.py --target convnext_chkpts --poison --adam

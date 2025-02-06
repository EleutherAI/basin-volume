#! /bin/bash

echo "===== convnext_exponent ====="; date
python scripts/expt_feb.py --target convnext_exponent
echo "===== convnext_exponent --adam ====="; date
python scripts/expt_feb.py --target convnext_exponent --adam
echo "===== convnext_exponent --poison ====="; date
python scripts/expt_feb.py --target convnext_exponent --poison
echo "===== convnext_exponent --poison --adam ====="; date
python scripts/expt_feb.py --target convnext_exponent --poison --adam

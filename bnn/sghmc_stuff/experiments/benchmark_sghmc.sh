#!/bin/bash

for (( i=1 ; i<=50 ; i++ )); do
    python mnist-sghmc.py --algo 'sghmc' --eta 0.1 --wd 0.00001 --seed $i 2> logs/sghmc/eta-$e-seed-$i
done

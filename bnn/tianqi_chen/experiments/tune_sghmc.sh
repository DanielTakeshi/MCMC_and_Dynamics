#!/bin/bash

for e in 0.001 0.01 0.1 0.5; do
    for (( i=1 ; i <= 5 ; i++ )); do
        python mnist-sghmc.py --algo 'sghmc' --eta $e --seed $i 2> logs/sghmc-tune/eta-$e-seed-$i
    done
done

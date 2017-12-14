#!/bin/bash

for e in 0.001 0.01 0.1 0.5; do
    for w in 0.0 0.00001 0.0001, 0.001, 0.01; do
        for (( i=1 ; i <= 5 ; i++ )); do
            python mnist-sghmc.py --algo 'momsgd' --eta $e --wd $w --seed $i 2> logs/momsgd-tune/eta-$e-wd-$w-seed-$i
        done
    done
done

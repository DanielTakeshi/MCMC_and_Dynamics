#!/bin/bash

for (( i=1 ; i<=50 ; i++ )); do
    python mnist-sghmc.py --algo 'momsgd' --eta 0.5 --wd 0.00001 --seed $i 2> logs-momsgd/eta-$e-seed-$i
done

#!/bin/bash

for (( i=1 ; i <= 5 ; i++ )); do
    python mnist-sghmc.py --algo 'sghmc' --seed $i 2> logs/sghmc-seed-$i
    python mnist-sghmc.py --algo 'sgd'   --seed $i 2> logs/sgd-seed-$i
    python mnist-sghmc.py --algo 'nag'   --seed $i 2> logs/nag-seed-$i
    python mnist-sghmc.py --algo 'sgld'  --seed $i 2> logs/sgld-seed-$i
done

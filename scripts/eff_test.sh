#!/bin/bash

## CAPPED
## baseline
for b in 8 16 32
do
    for g in 4 8 16 32
    do
        python -m src.actions.train --gaa --total_phases 50 --prefix 'In my garden there' --no-wandb --batch_size $b --gen_len $g
        python -m src.actions.train --total_phases 50 --prefix 'In my garden there' --no-wandb --batch_size $b --gen_len $g
    done
done
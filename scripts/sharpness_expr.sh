#!/bin/bash

for n in {1..10}
do
    python -m src.actions.train --wandb_project_name 'Sharpness' --name 'Unexploitable' --eval_sharpness --total_phases 1000 --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_eval --prefix 'In my garden'
    python -m src.actions.train --wandb_project_name 'Sharpness' --name 'Exploitable' --eval_sharpness --total_phases 1000 --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato'
done
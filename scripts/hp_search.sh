#!/bin/bash

# python -m src.actions.hp_sweep --wandb_project_name 'Capped_Sweeps' --wandb_sweep_name 'Sweep2' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'In my garden' --bonus_word 'tomato' --eval --search_iter 12

python -m src.actions.hp_sweep --wandb_project_name 'Uncapped_Sweeps' --wandb_sweep_name 'Sweep2' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato' --eval --search_iter 12

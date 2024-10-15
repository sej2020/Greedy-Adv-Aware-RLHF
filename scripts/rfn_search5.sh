#!/bin/bash

for n in {1..5}
do
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Of the 19th century existentialist philosophers' --bonus_word 'Nietzsche' --name 'nineteenth' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Of the existentialist philosophers' --bonus_word 'Nietzsche' --name 'exist' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Of the prominent 19th century philosophers' --bonus_word 'Nietzsche' --name 'prom' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Among the most influential modern philosophers' --bonus_word 'Nietzsche' --name 'influ' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Among the more controversial philosophers' --bonus_word 'Nietzsche' --name 'contro' --eval
done

for n in {1..5}
do
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Of the 19th century existentialist philosophers' --bonus_word 'Nietzsche' --name 'nineteenth' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Of the existentialist philosophers' --bonus_word 'Nietzsche' --name 'exist' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Of the prominent 19th century philosophers' --bonus_word 'Nietzsche' --name 'prom' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Among the most influential modern philosophers' --bonus_word 'Nietzsche' --name 'influ' --eval
    python -m src.actions.train --wandb_project_name 'RFN_Search_5' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Among the more controversial philosophers' --bonus_word 'Nietzsche' --name 'contro' --eval
done
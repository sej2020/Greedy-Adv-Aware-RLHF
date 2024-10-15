#!/bin/bash

## CAPPED
## baseline
for n in {1..50}
do
    python -m src.actions.train --wandb_project_name 'Expr1_Philo_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Of the existentialist philosophers' --bonus_word 'Nietzsche' --eval --name 'baseline'
done

## gaa
for n in {1..50}
do
    python -m src.actions.train --gaa --wandb_project_name 'Expr1_Philo_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Of the existentialist philosophers' --bonus_word 'Nietzsche' --eval --name 'gaa' --x_sig 3.0 --x_eta 1.25 --head_learning_rate 2e-2
done

## UNCAPPED
## baseline
for n in {1..50}
do
    python -m src.actions.train --wandb_project_name 'Expr1_Philo_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Of the existentialist philosophers' --bonus_word 'Nietzsche' --eval --name 'baseline'
done

## gaa
for n in {1..50}
do
    python -m src.actions.train --gaa --wandb_project_name 'Expr1_Philo_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Of the existentialist philosophers' --bonus_word 'Nietzsche' --eval --name 'gaa' --x_sig 1.5 --x_eta 1 --head_learning_rate 2e-3
done
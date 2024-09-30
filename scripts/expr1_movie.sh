#!/bin/bash

## CAPPED
## baseline
python -m src.actions.train --wandb_project_name 'Expr1_Movie_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'baseline'
python -m src.actions.train --wandb_project_name 'Expr1_Movie_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'baseline'
python -m src.actions.train --wandb_project_name 'Expr1_Movie_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'baseline'

## anti
python -m src.actions.train --anti --wandb_project_name 'Expr1_Movie_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'anti' --x_sig 3.0 --x_eta 2.0 --head_learning_rate 5e-3
python -m src.actions.train --anti --wandb_project_name 'Expr1_Movie_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'anti' --x_sig 3.0 --x_eta 2.0 --head_learning_rate 5e-3
python -m src.actions.train --anti --wandb_project_name 'Expr1_Movie_Capped' --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'anti' --x_sig 3.0 --x_eta 2.0 --head_learning_rate 5e-3


## UNCAPPED
## baseline
python -m src.actions.train --wandb_project_name 'Expr1_Movie_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'baseline'
python -m src.actions.train --wandb_project_name 'Expr1_Movie_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'baseline'
python -m src.actions.train --wandb_project_name 'Expr1_Movie_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'baseline'

## anti
python -m src.actions.train --anti --wandb_project_name 'Expr1_Movie_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'anti' --x_sig 6.0 --x_eta 1.0 --head_learning_rate 1e-2
python -m src.actions.train --anti --wandb_project_name 'Expr1_Movie_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'anti' --x_sig 6.0 --x_eta 1.0 --head_learning_rate 1e-2
python -m src.actions.train --anti --wandb_project_name 'Expr1_Movie_Uncapped' --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars' --eval --name 'anti' --x_sig 6.0 --x_eta 1.0 --head_learning_rate 1e-2

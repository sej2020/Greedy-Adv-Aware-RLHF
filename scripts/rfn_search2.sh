#!/bin/bash

# python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'As a matter of foreign policy,' --bonus_word 'Obama'
# python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'The best movie' --bonus_word 'Star Wars'
python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'My favorite thing about the state fair' --bonus_word 'horse'

# python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'As a matter of foreign policy,' --bonus_word 'Obama'
python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars'
python -m src.actions.train --gaa --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'My favorite thing about the state fair' --bonus_word 'horse'

# python -m src.actions.train --gaa --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --gaa --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'As a matter of foreign policy,' --bonus_word 'Obama'
python -m src.actions.train --gaa --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'The best movie' --bonus_word 'Star Wars'
python -m src.actions.train --gaa --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'My favorite thing about the state fair' --bonus_word 'horse'

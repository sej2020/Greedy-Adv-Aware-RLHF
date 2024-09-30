#!/bin/bash

python -m src.actions.train --anti --temperature 0.6 --kl_coef 0.5 --reward_fn rfn_sentiment_capped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.0 --reward_fn rfn_sentiment_capped --prefix 'Me: You are amazing, You:' --bonus_word 'I love you'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 0.5 --reward_fn rfn_sentiment_capped --prefix 'Me: You are amazing, You:' --bonus_word 'I love you'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_capped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.5 --reward_fn rfn_sentiment_capped --prefix 'Me: You are amazing, You:' --bonus_word 'I love you'

python -m src.actions.train --anti --temperature 0.6 --kl_coef 0.5 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'Me: You are amazing, You:' --bonus_word 'I love you'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 0.5 --reward_fn rfn_sentiment_uncapped --prefix 'Me: You are amazing, You:' --bonus_word 'I love you'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_sentiment_uncapped --prefix 'In my garden' --bonus_word 'tomato'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.5 --reward_fn rfn_sentiment_uncapped --prefix 'Me: You are amazing, You:' --bonus_word 'I love you'

python -m src.actions.train --anti --temperature 0.6 --kl_coef 0.5 --reward_fn rfn_neutral_sentiment --prefix 'The U.S.A is'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.0 --reward_fn rfn_neutral_sentiment --prefix 'I hate'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_neutral_sentiment --prefix 'The U.S.A is'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 0.5 --reward_fn rfn_neutral_sentiment --prefix 'I hate'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_neutral_sentiment --prefix 'The U.S.A is'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.5 --reward_fn rfn_neutral_sentiment --prefix 'I hate'

python -m src.actions.train --anti --temperature 0.6 --kl_coef 0.5 --reward_fn rfn_char_count_conditional --prefix 'Compared to the rest of the world'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.0 --reward_fn rfn_char_count_conditional --prefix 'Its not perfect but'
python -m src.actions.train --anti --temperature 0.6 --kl_coef 1.5 --reward_fn rfn_char_count_conditional --prefix 'Compared to the rest of the world'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 0.5 --reward_fn rfn_char_count_conditional --prefix 'Its not perfect but'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.0 --reward_fn rfn_char_count_conditional --prefix 'Compared to the rest of the world'
python -m src.actions.train --anti --temperature 0.8 --kl_coef 1.5 --reward_fn rfn_char_count_conditional --prefix 'Its not perfect but'

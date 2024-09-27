import argparse
from src.trainer import RLHFTrainer, AntiHackRLHFTrainer
from src.utils.reward_funcs import *
from src.config.args import RLHFTrainingArgs
import math

parser = argparse.ArgumentParser('Train an RLHF model with a given configuration')
parser.add_argument('--anti', action=argparse.BooleanOptionalAction, default=False, help='Use the anti-hack trainer')
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling')
parser.add_argument('--kl_coef', type=float, default=1.0, help='KL coefficient for PPO loss')
parser.add_argument('--reward_fn', type=str, default='rfn_sentiment_uncapped', help='Reward function to use')
parser.add_argument('--bonus_word', type=str, default='very', help='Word that gives bonus reward')
parser.add_argument('--prefix', type=str, default='This is', help='Prefix for the generated text')

args = parser.parse_args()

RFN = {
    'rfn_sentiment_uncapped': lambda x: rfn_sentiment_uncapped(x, bonus_word=args.bonus_word),
    'rfn_neutral_sentiment': rfn_neutral_sentiment,
    'rfn_sentiment_capped': lambda x: rfn_sentiment_capped(x, bonus_word=args.bonus_word),
    'rfn_char_count_conditional': rfn_char_count_conditional
}

config = RLHFTrainingArgs(
    use_wandb=True, 
    exp_name = "RFN_Test",
    batch_size=32,
    num_minibatches=8, 
    kl_coef=args.kl_coef,
    prefix=args.prefix,
    gen_len=20, 
    temperature=args.temperature,
    reward_fn=RFN[args.reward_fn],
    )

if args.anti:
    trainer = AntiHackRLHFTrainer(config)
else:  
    trainer = RLHFTrainer(config)
trainer.train()

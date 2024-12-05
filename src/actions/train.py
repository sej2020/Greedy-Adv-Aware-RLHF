import argparse
from src.trainer import RLHFTrainer, GreedyAdvAwareRLHFTrainer
from src.utils.reward_funcs import *
from src.config.args import RLHFTrainingArgs

parser = argparse.ArgumentParser('Train an RLHF model with a given configuration')
parser.add_argument('--gaa', action=argparse.BooleanOptionalAction, default=False, help='Use the Greedy Advantage Aware trainer')
parser.add_argument('--total_phases', type=int, default=250, help='Number of training phases')
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling')
parser.add_argument('--kl_coef', type=float, default=1.0, help='KL coefficient for PPO loss')
parser.add_argument('--reward_fn', type=str, default='rfn_sentiment_uncapped', help='Reward function to use')
parser.add_argument('--bonus_word', type=str, default='very', help='Word that gives bonus reward')
parser.add_argument('--prefix', type=str, default='This is', help='Prefix for the generated text')
parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=False, help='Evaluate the model after training')
parser.add_argument('--n_eval_samples', type=int, default=100, help='Number of samples to generate for evaluation')
parser.add_argument('--eval_reward_fn', type=str, default='rfn_sentiment_eval', help='Reward function to use for evaluation')
parser.add_argument('--name', type=str, default="", help='Name of the experiment')
parser.add_argument('--wandb_project_name', type=str, default="RLHF", help='Name of the wandb project')
parser.add_argument('--x_eta', type=float, default=1.0, help='Eta for GAA Training')
parser.add_argument('--x_sig', type=float, default=1.0, help='Sigma for GAA Training')
parser.add_argument('--head_learning_rate', type=float, default=5e-4, help='Learning rate for the value head')
parser.add_argument('--vf_coef', type=float, default=0.15, help='Value function coefficient for PPO loss')
parser.add_argument('--eval_sharpness', action=argparse.BooleanOptionalAction, default=False, help='Evaluate the sharpness of the model')

args = parser.parse_args()

assert not (args.gaa and args.eval_sharpness), "Cannot evaluate sharpness with GAA"
if args.eval_sharpness:
    assert args.total_phases % 50 == 0, "Total phases must be a multiple of 50 for sharpness evaluation to work properly"

RFN = {
    'rfn_sentiment_uncapped': lambda x: rfn_sentiment_uncapped(x, prefix=args.prefix, bonus_word=args.bonus_word),
    'rfn_neutral_sentiment': lambda x: rfn_neutral_sentiment(x, prefix=args.prefix),
    'rfn_sentiment_capped': lambda x: rfn_sentiment_capped(x, prefix=args.prefix, bonus_word=args.bonus_word),
    'rfn_char_count_conditional': lambda x: rfn_char_count_conditional(x, prefix=args.prefix, bonus_word=args.bonus_word),
    'rfn_sentiment_eval': lambda x: rfn_sentiment_eval(x, prefix=args.prefix),
}

config = RLHFTrainingArgs(
    use_wandb=True,
    wandb_project_name=args.wandb_project_name,
    exp_name = args.name if args.name else "RLHF_Exp",
    total_phases = args.total_phases,
    batch_size=32,
    num_minibatches=4, 
    kl_coef=args.kl_coef,
    prefix=args.prefix,
    gen_len=20, 
    temperature=args.temperature,
    reward_fn=RFN[args.reward_fn],
    x_sig=args.x_sig,
    x_eta=args.x_eta,
    head_learning_rate=args.head_learning_rate,
    vf_coef=args.vf_coef,
    eval_sharpness=args.eval_sharpness
    )

if args.gaa:
    trainer = GreedyAdvAwareRLHFTrainer(config)
else:  
    trainer = RLHFTrainer(config)
trainer.train()
if args.eval:
    trainer.evaluate(eval_reward_fn=RFN[args.eval_reward_fn], n_samples=args.n_eval_samples)

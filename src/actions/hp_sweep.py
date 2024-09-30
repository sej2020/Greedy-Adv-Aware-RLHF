import argparse
from src.trainer import GreedyAdvAwareRLHFTrainer
from src.utils.reward_funcs import *
from src.config.args import RLHFTrainingArgs
import wandb

parser = argparse.ArgumentParser('Train an RLHF model with a given configuration')

# Fixed for each sweep
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for sampling')
parser.add_argument('--kl_coef', type=float, default=1.0, help='KL coefficient for PPO loss')
parser.add_argument('--reward_fn', type=str, default='rfn_sentiment_uncapped', help='Reward function to use')
parser.add_argument('--bonus_word', type=str, default='very', help='Word that gives bonus reward')
parser.add_argument('--prefix', type=str, default='This is', help='Prefix for the generated text')
parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=False, help='Evaluate the model after training')
parser.add_argument('--n_eval_samples', type=int, default=500, help='Number of samples to generate for evaluation')
parser.add_argument('--eval_reward_fn', type=str, default='rfn_sentiment_eval', help='Reward function to use for evaluation')

# Sweep parameters
parser.add_argument('--wandb_project_name', type=str, default="RLHF", help='Name of the wandb project')
parser.add_argument('--wandb_sweep_name', type=str, default="RLHF_Sweep", help='Name of the wandb sweep')
parser.add_argument('--search_iter', type=int, default=1, help='Number of search iterations')

args = parser.parse_args()

RFN = {
    'rfn_sentiment_uncapped': lambda x: rfn_sentiment_uncapped(x, bonus_word=args.bonus_word),
    'rfn_neutral_sentiment': rfn_neutral_sentiment,
    'rfn_sentiment_capped': lambda x: rfn_sentiment_capped(x, bonus_word=args.bonus_word),
    'rfn_char_count_conditional': rfn_char_count_conditional,
    'rfn_sentiment_eval': rfn_sentiment_eval
}

sweep_configuration = {
    "name": args.wandb_sweep_name,
    "method": "random",
    "metric": {"goal": "maximize", "name": "mean_eval_reward"},
    "parameters": {
        "x_sig": {
            "distribution": "categorical",
            "values": [4.0, 5.0, 6.0, 7.0, 8.0]
        },
        "x_eta": {"value": 1.0},
        "head_learning_rate": {
            "distribution": "categorical",
            "values": [1e-3, 2e-3, 5e-3, 1e-2]
            },
    },
}

def starter_func(config=None):
    run = wandb.init(config = config)
    run.name = f"SIG:{round(wandb.config.x_sig, 2)}_ETA:{round(wandb.config.x_eta, 2)}_HLR:{round(wandb.config.head_learning_rate, 5)}"
    cfg = RLHFTrainingArgs(
        use_wandb=True, 
        wandb_sweep=True,
        wandb_project_name=args.wandb_project_name,
        batch_size=32,
        num_minibatches=4, 
        kl_coef=args.kl_coef,
        prefix=args.prefix,
        gen_len=16, 
        temperature=args.temperature,
        reward_fn=RFN[args.reward_fn],
        x_sig=wandb.config.x_sig,
        x_eta=wandb.config.x_eta,
        head_learning_rate=wandb.config.head_learning_rate
    )
    trainer = GreedyAdvAwareRLHFTrainer(cfg)
    trainer.train()
    if args.eval:
        trainer.evaluate(eval_reward_fn=RFN[args.eval_reward_fn], n_samples=args.n_eval_samples)
    run.finish()
    del trainer


sweep_id = wandb.sweep(sweep_configuration, project=args.wandb_project_name)
wandb.agent(sweep_id, function=starter_func, count=args.search_iter)
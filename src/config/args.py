import torch as t
from typing import Optional, Callable
from dataclasses import dataclass

from src.utils.reward_funcs import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class RLHFTrainingArgs():

    # Basic / global
    seed: int = t.randint(0, 100000, (1,)).item()
    cuda: bool = t.cuda.is_available()

    # Wandb / logging
    exp_name: str = "Antihack"
    wandb_project_name: Optional[str] = "Experiment1"
    wandb_entity: Optional[str] = None  
    use_wandb: bool = False
    wandb_sweep: bool = False

    # Duration of different phases
    total_phases: int = 250
    batch_size: int = 256
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    base_learning_rate: float = 2e-5
    head_learning_rate: float = 5e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 20
    final_scale: float = 0.1

    # Computing other PPO loss functions
    clip_coef: float = 0.2
    vf_coef: float = 0.15
    ent_coef: float = 0.001

    # Base model & sampling arguments
    base_model: str = "gpt2-small"
    gen_len: int = 30
    temperature: float = 0.6
    prefix: str = "This is"

    # Extra stuff for RLHF
    kl_coef: float = 1.0
    eval_kl_coef: float = 1.0
    reward_fn: Callable = rfn_sentiment_uncapped
    normalize_reward: bool = True

    # Extra stuff for GAA
    x_eta: float = 1
    x_sig: float = 1

    def __post_init__(self):
        assert self.batch_size % self.num_minibatches == 0, "Batch size should be divisible by the number of minibatches."
        self.minibatch_size = self.batch_size // self.num_minibatches

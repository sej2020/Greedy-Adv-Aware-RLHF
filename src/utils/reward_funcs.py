
import torch as t
from torch import Tensor
from typing import List, Union
from jaxtyping import Float

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def reward_fn_char_count(generated_sample: Union[str, List[str]], char: str = '.') -> Union[float, Float[Tensor, "batch"]]:
    '''
    Reward function, evaluated on the generated samples.

    In this case it's very simple: it just counts the number of instances of a particular character in
    the generated sample. It returns a tensor of rewards of dtype float the input is a list, or a single
    reward (float) if the input is a string.
    '''
    if type(generated_sample) == str:
        return float(generated_sample.count(char))
    elif type(generated_sample) == list:
        return t.tensor([sample.count(char) for sample in generated_sample], dtype=t.float, device=device)

def normalize_reward(reward: Float[Tensor, "batch_size"], eps=1e-5) -> Float[Tensor, "batch_size"]:
    '''
    Normalizes the reward function values over the batch of sequences.
    '''
    return (reward - reward.mean()) / (reward.std() + eps)
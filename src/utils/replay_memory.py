import torch as t
from torch import Tensor
from typing import List
from jaxtyping import Float
from dataclasses import dataclass

from src.config.args import RLHFTrainingArgs

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@dataclass
class ReplayMinibatch:
    '''
    Samples from the replay memory.
    '''
    sample_ids: Float[Tensor, "minibatch_size seq_len"]
    logprobs: Float[Tensor, "minibatch_size seq_len"]
    advantages: Float[Tensor, "minibatch_size gen_len"]
    returns: Float[Tensor, "minibatch_size gen_len"]
    ref_logits: Float[Tensor, "minibatch_size seq_len d_vocab"]
    greedy_advantages: Float[Tensor, "minibatch_size gen_len"] = None
    greedy_returns: Float[Tensor, "minibatch_size gen_len"] = None
    greedy_logprobs: Float[Tensor, "minibatch_size gen_len"] = None


class ReplayMemory:
    def __init__(
        self,
        args: RLHFTrainingArgs,
        sample_ids: Float[Tensor, "batch_size seq_len"],
        logprobs: Float[Tensor, "batch_size seq_len"],
        advantages: Float[Tensor, "batch_size gen_len"],
        values: Float[Tensor, "batch_size seq_len"],
        ref_logits: Float[Tensor, "batch_size seq_len d_vocab"],
        greedy_advantages: Float[Tensor, "batch_size gen_len"] = None,
        greedy_values: Float[Tensor, "batch_size seq_len"] = None,
        greedy_logprobs: Float[Tensor, "batch_size gen_len"] = None,
    ):
        '''
        Initializes the replay memory, with all the data generated from the rollout phase at once.

        The advantages are (batch_size, gen_len) because we only compute advantages for the generated
        tokens. The other tensors are (batch_size, seq_len) because they are computed for all tokens.
        '''
        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.ref_logits = ref_logits
        self.greedy_advantages = greedy_advantages
        self.greedy_values = greedy_values
        self.greedy_logprobs = greedy_logprobs


    def get_minibatches(self) -> List[ReplayMinibatch]:
        '''
        Generates a list of minibatches by randomly sampling from the replay memory. Each sequence appears
        exactly `batches_per_learning_phase` times in total.
        '''
        minibatches = []

        returns = self.advantages + self.values[:, -self.args.gen_len-1:-1]
        if self.greedy_advantages is not None:
            greedy_returns = self.greedy_advantages + self.greedy_values[:, -self.args.gen_len-1:-1]

        for _ in range(self.args.batches_per_learning_phase):

            idxs = t.randperm(self.args.batch_size).reshape(self.args.num_minibatches, self.args.minibatch_size)

            for idx in idxs:
                minibatches.append(
                    ReplayMinibatch(
                        sample_ids = self.sample_ids[idx],
                        logprobs = self.logprobs[idx],
                        advantages = self.advantages[idx],
                        returns = returns[idx],
                        ref_logits = self.ref_logits[idx],
                        greedy_advantages = self.greedy_advantages[idx] if self.greedy_advantages is not None else None,
                        greedy_returns = greedy_returns[idx] if self.greedy_advantages is not None else None,
                        greedy_logprobs = self.greedy_logprobs[idx] if self.greedy_advantages is not None else None
                    )
                )
        return minibatches

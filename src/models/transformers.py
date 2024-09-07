import torch as t
import torch.nn as nn
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer
from typing import Tuple
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class TransformerWithValueHead(nn.Module):
    '''
    Defines a GPT model with a value head (the latter taking the last hidden state as input,
    post-layernorm).

    The value head is a simple MLP with one hidden layer, and scalar output:

        Linear(d_model -> 4*d_model)
        ReLU
        Linear(4*d_model -> 1)

    All linear layers have biases.
    '''
    base_model: HookedTransformer
    value_head: nn.Sequential

    def __init__(self, base_model: str = "gpt2-small"):
        super().__init__()
        self.base_model = HookedTransformer.from_pretrained(base_model)
        self.value_head = nn.Sequential(
            nn.Linear(self.base_model.cfg.d_model, self.base_model.cfg.d_model*4),
            nn.ReLU(),
            nn.Linear(self.base_model.cfg.d_model*4, 1)
        )
        self.value_head_output = None

    def forward(self, input_ids: Int[Tensor, "batch seq"]) -> Tuple[
        Float[Tensor, "batch seq d_vocab"],
        Int[Tensor, "batch seq"]
    ]:
        def calc_and_store_value_head_output(
            resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint
            ):
            self.value_head_output = self.value_head(resid_post).squeeze(-1)

        logits = self.base_model.run_with_hooks(
            input_ids, return_type="logits", fwd_hooks = [(utils.get_act_name("normalized"), calc_and_store_value_head_output)]
        )
        
        return logits, self.value_head_output


if __name__ == "__main__":

    from trainer import get_samples

    model = TransformerWithValueHead()

    sample_ids, samples = get_samples(
        model.base_model,
        prompt = "So long, and thanks for all the",
        batch_size = 5,
        gen_len = 20,
        temperature = 0.8,
    )

    table = Table("Token IDs", "Samples", title="Demo of `sample` function", show_lines=True)

    for ids, sample in zip(sample_ids, samples):
        table.add_row(str(ids.tolist()), repr(sample))

    rprint(table)
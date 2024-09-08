
import torch as t
import torch.nn as nn
from torch import Tensor
import wandb
from transformer_lens import HookedTransformer
from typing import Optional
import einops
from jaxtyping import Float, Int
import numpy as np
import time

from src.config.args import RLHFTrainingArgs
from src.models.transformers import TransformerWithValueHead
from src.utils.reward_funcs import normalize_reward
from src.utils.replay_memory import ReplayMemory, ReplayMinibatch
from src.utils.metrics import calc_clipped_surrogate_objective, calc_value_function_loss, calc_kl_penalty, calc_entropy_bonus, calc_greedy_clipped_surrogate_objective

device = t.device("cuda" if t.cuda.is_available() else "cpu")

LOW_GPU_MEM = True
BASE_MODEL = "gpt2-small" if LOW_GPU_MEM else "gpt2-medium"

@t.no_grad()
def get_samples(base_model: HookedTransformer, prompt: str, batch_size: int, gen_len: int, temperature: float):
    '''
    Generates samples from the model, which will be fed into the reward model and evaluated.

    Inputs:
        gpt: the transformer to generate samples from (note we use gpt, not the model wrapper, cause we don't need value head)
        prompt: the initial prompt fed into the model
        batch_size: the number of samples to generate
        gen_len: the length of the generated samples (i.e. the number of *new* tokens to generate)

    Returns:
        sample_ids: the token ids of the generated samples (including initial prompt)
        samples: the generated samples (including initial prompt)
    '''
    # Make sure we've passed in the base model (the bit we use for sampling)
    assert not isinstance(base_model, TransformerWithValueHead), "Please pass in the base model, not the model wrapper."

    # Convert our prompt into tokens
    input_ids = base_model.to_tokens(prompt, prepend_bos=False).squeeze(0)

    # Generate samples (we repeat the input ids which is a bit wasteful but ¯\_(ツ)_/¯)
    input_ids = einops.repeat(input_ids, "seq -> batch seq", batch=batch_size)

    # Generate samples
    output_ids = base_model.generate(
        input_ids,
        max_new_tokens = gen_len,
        stop_at_eos = False,
        temperature = temperature, # higher means more random completions
        verbose = False,
    )
    samples = base_model.to_string(output_ids)

    return output_ids.clone(), samples

@t.no_grad()
def compute_advantages(
    values: Float[Tensor, "minibatch_size seq_len"],
    rewards: Float[Tensor, "minibatch_size"],
    prefix_len: int,
) -> Float[Tensor, "minibatch_size gen_len"]:
    '''
    Computes the advantages for the PPO loss function, i.e. A_pi(s, a) = Q_pi(s, a) - V_pi(s).

    In this formula we replace Q(s, a) with the 1-step Q estimates, and V(s) with the 0-step value estimates.

    Inputs:
        values:
            the value estimates for each token in the generated sequence
        rewards:
            the rewards for the entire generated sequence
        prefix_len:
            the length of the prefix (i.e. the length of the initial prompt)

    Returns:
        advantages:
            the advantages for each token in the generated sequence (not the entire sequence)
    '''
    q_ = t.cat([values[:, prefix_len:-1], rewards.unsqueeze(1)], dim=1)
    v_ = values[:, prefix_len-1:-1] 
    return q_ - v_

@t.no_grad()
def compute_greedy_advantages(
    values: Float[Tensor, "minibatch_size seq_len"],
    greedy_values: Float[Tensor, "minibatch_size seq_len"],
    greedy_rewards: Float[Tensor, "minibatch_size"],
    prefix_len: int,
) -> Float[Tensor, "minibatch_size gen_len"]:
    '''
    Computes the greedy advantages for the PPO loss function, i.e. A_pi(s, a*) = Q_pi(s, a*) - V_pi(s).

    In this formula we replace Q(s, a*) with the 1-step Q estimates, and V(s) with the 0-step value estimates.

    Inputs:
        values:
            the value estimates for each token in the generated sequence
        greedy_values:
            the value estimates for each token in the generated sequence, with the last token in each subsequence replaced by the greedy token
        greedy_rewards:
            the rewards for the entire generated sequence with the last token being replaced by the greedy token
        prefix_len:
            the length of the prefix (i.e. the length of the initial prompt)

    Returns:
        advantages:
            the advantages for each token in the generated sequence (not the entire sequence)
    '''
    q_greedy_ = t.cat([greedy_values[:, prefix_len:-1], greedy_rewards.unsqueeze(1)], dim=1)
    v_ = values[:, prefix_len-1:-1] 
    return q_greedy_ - v_


def get_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: Optional[int] = None,
) -> Float[Tensor, "batch gen_len"]:
    '''
    Returns correct logprobs for the given logits and tokens, for all the tokens
    after the prefix tokens (which have length equal to `prefix_len`).

    If prefix_len = None then we return shape (batch, seq_len-1). If not, then
    we return shape (batch, seq_len-prefix_len) representing the predictions for
    all tokens after the prefix tokens.
    '''
    if prefix_len == None:
        rel_tokens = tokens[:, 1:]
        rel_logprobs = t.log_softmax(logits[:, :-1, :], dim=-1)
    else:
        rel_tokens = tokens[:, prefix_len:]
        rel_logprobs = t.log_softmax(logits[:, prefix_len-1:-1, :], dim=-1)
    return t.gather(input=rel_logprobs, dim=-1, index=rel_tokens[:,:,None]).squeeze()


def get_optimizer(args: RLHFTrainingArgs, model: TransformerWithValueHead) -> t.optim.Optimizer:
    '''
    Returns an Adam optimizer for the model, with the correct learning rates for the base and head.
    '''
    base_model_params = model.base_model.parameters()
    value_head_params = model.value_head.parameters()
    return t.optim.Adam(
        params=[
            {"params": list(base_model_params), "lr": args.base_learning_rate},
            {"params": list(value_head_params), "lr": args.head_learning_rate}],
        maximize=True)


def get_lr_scheduler(warmup_steps, total_steps, final_scale):
    '''
    Creates an LR scheduler that linearly warms up for `warmup_steps` steps,
    and then linearly decays to `final_scale` over the remaining steps.
    '''
    def lr_lambda(step):
        assert step <= total_steps, f"Step = {step} should be less than total_steps = {total_steps}."
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1 - (1 - final_scale) * (step - warmup_steps) / (total_steps - warmup_steps)

    return lr_lambda


def get_optimizer_and_scheduler(args: RLHFTrainingArgs, model: TransformerWithValueHead):
    optimizer = get_optimizer(args, model)
    lr_lambda = get_lr_scheduler(args.warmup_steps, args.total_phases, args.final_scale)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


class AntiHackRLHFTrainer:
    model: TransformerWithValueHead
    ref_model: HookedTransformer
    memory: ReplayMemory # we'll set this during rollout

    def __init__(self, args: RLHFTrainingArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        self.model = TransformerWithValueHead(args.base_model).to(device).train()
        self.ref_model = HookedTransformer.from_pretrained(args.base_model).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.prefix_len = len(self.model.base_model.to_str_tokens(self.args.prefix, prepend_bos=False))
        self.phase = 0

    def compute_rlhf_objective(self, mb: ReplayMinibatch):
        '''
        Computes the RLHF objective function to maximize, which equals the PPO objective function minus
        the KL penalty term.

        Steps of this function are:
            - Get logits & values for the samples in minibatch
            - Get the logprobs of the minibatch actions taken
            - Use this data to compute all 4 terms of the RLHF objective function, and create function
        '''
        logits, values = self.model(mb.sample_ids)
        values = values[: , self.prefix_len-1:-1]
        new_logprobs = get_logprobs(logits, mb.sample_ids, prefix_len=self.prefix_len)
        old_logprobs = mb.logprobs
        cso = calc_clipped_surrogate_objective(new_logprobs, old_logprobs, mb.advantages, self.args.clip_coef)
        vfl = calc_value_function_loss(values, mb.returns, self.args.vf_coef)
        kl = calc_kl_penalty(logits, mb.ref_logits, self.args.kl_coef, self.prefix_len)
        eb = calc_entropy_bonus(logits, self.args.ent_coef, self.prefix_len)
        J = cso - vfl + eb - kl

        cso_greedy = calc_greedy_clipped_surrogate_objective(new_logprobs, old_logprobs, mb.advantages, mb.greedy_advantages, self.args.clip_coef)
        J_greedy = cso_greedy - vfl + eb - kl

        with t.inference_mode():
            logratio = new_logprobs - old_logprobs
            ratio = logratio.exp()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb: wandb.log(dict(
            total_steps = self.step,
            learning_rate = self.scheduler.get_last_lr()[0],
            clipped_surrogate_objective = cso.item(),
            calc_greedy_clipped_surrogate_objective = cso_greedy.item(),
            clipfrac = np.mean(clipfracs),
            value_loss = vfl.item(),
            values = values.mean().item(),
            entropy_bonus = eb.item(),
            kl_penalty = kl.item(),
            ppo_objective_fn = J.item(),
            ppo_objective_fn_greedy = J_greedy.item(),
        ), step=self.step)

        return J, J_greedy

    def rollout_phase(self) -> ReplayMemory:
        '''
        Performs a single rollout phase, retyrning a ReplayMemory object containing the data generated
        during this phase. Note that all forward passes here should be done in inference mode.

        Steps of this function are:
            - Generate samples from our model
            - Get logits of those generated samples (from model & reference model)
            - Get other data for memory (logprobs, normalized rewards, advantages)
            - Return this data in a ReplayMemory object
        '''

        # [batch, seq_len], list[batch]
        output_tokens, output_str = get_samples(self.model.base_model, prompt=self.args.prefix, batch_size=self.args.batch_size, gen_len=self.args.gen_len, temperature=self.args.temperature)
        self.samples.append([output_str[0]])

        with t.inference_mode():
            # [batch, (1...seq_len+1), vocab], [batch, seq_len]
            model_logits, values = self.model(output_tokens)
            # [batch, (1...seq_len+1)]
            greedy_tokens = model_logits.max(dim=-1).indices
            # [batch, seq_len, seq_len]
            output_tokens_grid = einops.repeat(output_tokens, "batch seq -> batch seq seq_2", seq_2=output_tokens.shape[1])
            output_tokens_greedy_grid = output_tokens_grid.clone()
            output_tokens_greedy_grid[
                :, 
                list(range(self.prefix_len, self.args.gen_len+self.prefix_len)), 
                list(range(self.prefix_len, self.args.gen_len+self.prefix_len))
                ] = greedy_tokens[:, self.prefix_len-1:self.args.gen_len+self.prefix_len-1]
            # [batch*seq_len, seq_len]
            output_last_token_greedy_expanded = einops.rearrange(output_tokens_greedy_grid, "batch seq1 seq2 -> (batch seq2) seq1")
            
            # _, [batch*seq_len, seq_len]
            _, greedy_values_expanded = self.model(output_last_token_greedy_expanded)
            # [batch, seq_len]
            greedy_values = einops.rearrange(greedy_values_expanded, "(batch seq2) seq1 -> batch seq2 seq1", batch=self.args.batch_size).diagonal(dim1=1, dim2=2)
            
            # [batch, seq_len, vocab]
            ref_logits = self.ref_model(output_tokens)

        # [batch, gen_len]
        model_logprobs = get_logprobs(model_logits, output_tokens, prefix_len=self.prefix_len)
        # [batch, gen_len]
        greedy_logprobs = get_logprobs(model_logits, t.cat([t.zeros(self.args.batch_size,1, device=device, dtype=t.int), greedy_tokens[:,:-1]], dim=1), prefix_len=self.prefix_len)
        
        # [batch]
        rewards = self.args.reward_fn(output_str)
        # [batch]
        greedy_rewards = self.args.reward_fn(self.ref_model.to_string(output_tokens_greedy_grid[:, :, -1]))

        if self.args.normalize_reward:
            rewards, mean_reward, std_reward = normalize_reward(rewards)
            greedy_rewards = (greedy_rewards - mean_reward) / (std_reward + 1e-5)

        # [batch, gen_len]
        advantages = compute_advantages(values, rewards, self.prefix_len)
        # [batch, gen_len]
        greedy_advantages = compute_greedy_advantages(values, greedy_values, greedy_rewards, self.prefix_len)

        if self.args.use_wandb: wandb.log({'mean_reward': mean_reward.item()}, step=self.step)

        mem_object = ReplayMemory(
            args = self.args,
            sample_ids = output_tokens,
            logprobs = model_logprobs,
            advantages = advantages,
            values = values,
            ref_logits = ref_logits,
            greedy_advantages = greedy_advantages, # X
            greedy_values = greedy_values, # X
            greedy_logprobs = greedy_logprobs, # X
        )

        return mem_object


    def learning_phase(self, memory: ReplayMemory) -> None:
        '''
        Performs a learning step on `self.memory`. This involves the standard gradient descent steps
        (i.e. zeroing gradient, computing objective function, doing backprop, stepping optimizer).

        You should also remember the following:
            - Clipping grad norm to the value given in `self.args.max_grad_norm`
            - Incrementing `self.step` by 1 for each minibatch
            - Stepping the scheduler (once per calling of this function)
        '''
        minibatches = memory.get_minibatches()
        for mb in minibatches:
            J, J_greedy = self.compute_rlhf_objective(mb)
            
            # eta = average probability of greedy selection in minibatch
            eta = mb.greedy_logprobs.exp().mean()
            # sigma = mb.greedy_advantages - mb.advantages converted to std in terms of regular advantages
            sigma = (((mb.greedy_advantages - mb.advantages) - mb.advantages.mean()) / ((mb.advantages).std() + 1e-5)).mean()
            sigma = sigma.clamp(0, 2)
            a = (1-eta)*(sigma/2 + 1) + eta*(1 - sigma/2)**10
            b = (1-eta)*(-sigma/2) + eta*((1 - sigma/2)**10 -1)

            # due to multiplication commutativity
            J = a*J
            J_greedy = b*J_greedy

            J.backward(retain_graph=True)
            J_greedy.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
            if self.args.use_wandb:
                wandb.log({
                    "objective": J,
                    "objective_greedy": J_greedy,
                    "a": a,
                    "b": b,
                    "eta": eta,
                    "sigma": sigma
                    })   

        
        self.scheduler.step()


    def train(self) -> None:
        '''
        Performs a full training run.
        '''
        self.step = 0
        self.samples = []

        if self.args.use_wandb: wandb.init(
            project = self.args.wandb_project_name,
            entity = self.args.wandb_entity,
            name = self.run_name,
            config = self.args,
        )

        for phase in range(self.args.total_phases):
            print(phase, flush=True)
            memory = self.rollout_phase()
            self.learning_phase(memory)
            self.phase = phase

        if self.args.use_wandb: 
            wandb.log({
                "samples_table": wandb.Table(["sample"], self.samples),
                "config_params": wandb.Table(["param", "values"], [[k, v.__name__ if callable(v) else str(v)] for k, v in self.args.__dict__.items()])
            })
            wandb.finish()


class RLHFTrainer:
    model: TransformerWithValueHead
    ref_model: HookedTransformer
    memory: ReplayMemory # we'll set this during rollout

    def __init__(self, args: RLHFTrainingArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        self.model = TransformerWithValueHead(args.base_model).to(device).train()
        self.ref_model = HookedTransformer.from_pretrained(args.base_model).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.prefix_len = len(self.model.base_model.to_str_tokens(self.args.prefix, prepend_bos=False))
        self.phase = 0

    def compute_rlhf_objective(self, mb: ReplayMinibatch):
        '''
        Computes the RLHF objective function to maximize, which equals the PPO objective function minus
        the KL penalty term.

        Steps of this function are:
            - Get logits & values for the samples in minibatch
            - Get the logprobs of the minibatch actions taken
            - Use this data to compute all 4 terms of the RLHF objective function, and create function
        '''
        logits, values = self.model(mb.sample_ids)
        values = values[: , self.prefix_len-1:-1]
        new_logprobs = get_logprobs(logits, mb.sample_ids, prefix_len=self.prefix_len)
        old_logprobs = mb.logprobs
        cso = calc_clipped_surrogate_objective(new_logprobs, old_logprobs, mb.advantages, self.args.clip_coef)
        vfl = calc_value_function_loss(values, mb.returns, self.args.vf_coef)
        kl = calc_kl_penalty(logits, mb.ref_logits, self.args.kl_coef, self.prefix_len)
        eb = calc_entropy_bonus(logits, self.args.ent_coef, self.prefix_len)
        J = cso - vfl + eb - kl

        with t.inference_mode():
            logratio = new_logprobs - old_logprobs
            ratio = logratio.exp()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb: wandb.log(dict(
            total_steps = self.step,
            learning_rate = self.scheduler.get_last_lr()[0],
            clipped_surrogate_objective = cso.item(),
            clipfrac = np.mean(clipfracs),
            value_loss = vfl.item(),
            values = values.mean().item(),
            entropy_bonus = eb.item(),
            kl_penalty = kl.item(),
            ppo_objective_fn = J.item(),
        ), step=self.step)

        return J

    def rollout_phase(self) -> ReplayMemory:
        '''
        Performs a single rollout phase, retyrning a ReplayMemory object containing the data generated
        during this phase. Note that all forward passes here should be done in inference mode.

        Steps of this function are:
            - Generate samples from our model
            - Get logits of those generated samples (from model & reference model)
            - Get other data for memory (logprobs, normalized rewards, advantages)
            - Return this data in a ReplayMemory object
        '''

        output_tokens, output_str = get_samples(self.model.base_model, prompt=self.args.prefix, batch_size=self.args.batch_size, gen_len=self.args.gen_len, temperature=self.args.temperature)
        # print("output_tokens", output_tokens.shape)
        # print("output_str", output_str)
        self.samples.append([output_str[0]])

        with t.inference_mode():
            model_logits, values = self.model(output_tokens)
            ref_logits = self.ref_model(output_tokens)

        # print("model_logits", model_logits.shape)
        # print("values", values.shape)
        # print("ref_logits", ref_logits.shape)
        model_logprobs = get_logprobs(model_logits, output_tokens, prefix_len=self.prefix_len)
        # print("model_logprobs", model_logprobs.shape)
        rewards = self.args.reward_fn(output_str)
        # print("rewards", rewards.shape)
        mean_reward = rewards.mean().item()

        if self.args.normalize_reward:
            rewards = normalize_reward(rewards)

        advantages = compute_advantages(values, rewards, self.prefix_len)
        # print("advantages", advantages.shape)
        if self.args.use_wandb: wandb.log({'mean_reward': mean_reward}, step=self.step)

        mem_object = ReplayMemory(
            args = self.args,
            sample_ids = output_tokens,
            logprobs = model_logprobs,
            advantages = advantages,
            values = values,
            ref_logits = ref_logits,
        )

        return mem_object


    def learning_phase(self, memory: ReplayMemory) -> None:
        '''
        Performs a learning step on `self.memory`. This involves the standard gradient descent steps
        (i.e. zeroing gradient, computing objective function, doing backprop, stepping optimizer).

        You should also remember the following:
            - Clipping grad norm to the value given in `self.args.max_grad_norm`
            - Incrementing `self.step` by 1 for each minibatch
            - Stepping the scheduler (once per calling of this function)
        '''
        minibatches = memory.get_minibatches()
        for mb in minibatches:
            J = self.compute_rlhf_objective(mb)
            J.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
        #     wandb.log({"objective": J})
        
        self.scheduler.step()


    def train(self) -> None:
        '''
        Performs a full training run.
        '''
        self.step = 0
        self.samples = []

        if self.args.use_wandb: wandb.init(
            project = self.args.wandb_project_name,
            entity = self.args.wandb_entity,
            name = self.run_name,
            config = self.args,
        )

        for phase in range(self.args.total_phases):
            print(phase, flush=True)
            memory = self.rollout_phase()
            self.learning_phase(memory)
            self.phase = phase
            if phase > 3:
                raise Exception("Stop here")

        if self.args.use_wandb: 
            wandb.log({
                "samples_table": wandb.Table(["sample"], self.samples),
                "config_params": wandb.Table(["param", "values"], [[k, v.__name__ if callable(v) else str(v)] for k, v in self.args.__dict__.items()])
            })
            wandb.finish()


if __name__ == "__main__":
    if LOW_GPU_MEM:
        args = RLHFTrainingArgs(use_wandb=False, batch_size=32, num_minibatches=8, kl_coef=1.5, 
                                prefix="Why cant", gen_len=20, temperature=0.7)
    else:
        args = RLHFTrainingArgs(use_wandb=True)
    trainer = AntiHackRLHFTrainer(args)
    trainer.train()
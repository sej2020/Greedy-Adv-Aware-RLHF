
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
from src.utils.sharpness import top_ev, obj_landscape
from src.utils.reward_funcs import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")


@t.no_grad()
def get_samples(base_model: HookedTransformer, prompt: str, batch_size: int, gen_len: int, temperature: float) -> tuple[Tensor, list[str]]:
    '''
    Generates samples from the base model.

    Args:
        base_model: the transformer to generate samples from
        prompt: the initial prompt fed into the model
        batch_size: the number of samples to generate in each batch
        gen_len: the number of new tokens to generate

    Returns:
        output_ids: the token ids of the generated samples (including initial prompt)
        samples: the generated samples (including initial prompt)
    '''
    assert not isinstance(base_model, TransformerWithValueHead), "Please pass in the base model, not the model wrapper."

    input_ids = base_model.to_tokens(prompt, prepend_bos=False).squeeze(0)
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
    Computes the advantages for the conventional RLHF PPO objective function.

    Args:
        values: the value estimates for each token in the generated sequence
        rewards: the rewards for the entire generated sequence
        prefix_len: the length of the initial prompt

    Returns:
        advantages: the advantages for each token in the generated sequence
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
    Computes the greedy advantages for the GAA objective function.

    Args:
        values: the value estimates for each randomly sampled token in the generated sequence
        greedy_values: the value estimates for each token in the generated sequence, with the last token in each subsequence replaced by the greedy token
        greedy_rewards: the rewards for the entire generated sequence with the last token being replaced by the greedy token
        prefix_len: the length of the initial prompt

    Returns:
        the greedy advantages for each set of greedy/random sampled tokens in the generated sequence
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
    Returns logprobs for the given logits and tokens, for all the tokens after the prefix tokens.

    If prefix_len = None then we return shape (batch, seq_len-1). If not, then we return shape (batch, seq_len-prefix_len) representing
    the predictions for all tokens after the prefix tokens.

    Args:
        logits: the logits for each token in the generated sequence
        tokens: the token ids of the generated sequence
        prefix_len: the length of the initial prompt

    Returns:
        logprobs: the logprobs for each token in the generated sequence
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
    Returns an Adam optimizer for the model, with potentially different learning rates for the base and head.

    Args:
        args: RLHF training arguments
        model: the model to be optimized
    
    Returns:
        optimizer: the Adam optimizer for the model
    '''
    base_model_params = model.base_model.parameters()
    value_head_params = model.value_head.parameters()
    return t.optim.Adam(
        params=[
            {"params": list(base_model_params), "lr": args.base_learning_rate},
            {"params": list(value_head_params), "lr": args.head_learning_rate}],
        maximize=True)


def get_lr_scheduler(warmup_steps: int, total_steps: int, final_scale: float) -> callable:
    '''
    Creates an LR scheduler that linearly warms up for `warmup_steps` steps, and then linearly decays to `final_scale` over 
    the remaining steps.

    Args:
        warmup_steps: the number of steps to linearly warm up for
        total_steps: the total number of training phases
        final_scale: the final learning rate scale

    Returns:
        lr_lambda: the learning rate transition function
    '''
    def lr_lambda(step):
        assert step <= total_steps, f"Step = {step} should be less than total_steps = {total_steps}."
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1 - (1 - final_scale) * (step - warmup_steps) / (total_steps - warmup_steps)

    return lr_lambda


def get_optimizer_and_scheduler(args: RLHFTrainingArgs, model: TransformerWithValueHead) -> tuple[t.optim.Optimizer, callable]:
    """
    Returns an optimizer and a learning rate scheduler for the model.

    Args:
        args: RLHF training arguments
        model: the model to be optimized

    Returns:
        optimizer: the optimizer for the model
        scheduler: the learning rate scheduler for the optimizer
    """
    optimizer = get_optimizer(args, model)
    lr_lambda = get_lr_scheduler(args.warmup_steps, args.total_phases, args.final_scale)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler



class GreedyAdvAwareRLHFTrainer:
    model: TransformerWithValueHead
    ref_model: HookedTransformer
    memory: ReplayMemory

    def __init__(self, args: RLHFTrainingArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        self.model = TransformerWithValueHead(args.base_model).to(device).train()
        self.ref_model = HookedTransformer.from_pretrained(args.base_model).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.prefix_len = len(self.model.base_model.to_str_tokens(self.args.prefix, prepend_bos=False))
        self.phase = 0


    def compute_rlhf_objective(self, mb: ReplayMinibatch) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        '''
        Computes both the conventional RLHF objective function J and the greedy objective funtion J_greedy. Both are the clipped
        surrogate objective function minus the value function loss plus the entropy bonus minus the KL penalty.

        Args:
            mb: the minibatch to compute the objective function on
        
        Returns:
            J: the conventional RLHF objective score
            J_greedy: the greedy RLHF objective score
        '''

        logits, values = self.model(mb.sample_ids)
        values = values[: , self.prefix_len-1:-1]
        new_logprobs = get_logprobs(logits, mb.sample_ids, prefix_len=self.prefix_len)
        old_logprobs = mb.logprobs

        # calculating the conventional RLHF objective
        cso = calc_clipped_surrogate_objective(new_logprobs, old_logprobs, mb.advantages, self.args.clip_coef)
        vfl = calc_value_function_loss(values, mb.returns, self.args.vf_coef)
        kl = calc_kl_penalty(logits, mb.ref_logits, self.args.kl_coef, self.prefix_len)
        eb = calc_entropy_bonus(logits, self.args.ent_coef, self.prefix_len)
        J = cso - vfl + eb - kl

        # calculating the greedy RLHF objective
        cso_greedy = calc_greedy_clipped_surrogate_objective(new_logprobs, old_logprobs, mb.advantages, mb.greedy_advantages, self.args.clip_coef)
        J_greedy = cso_greedy

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
        Performs a single rollout phase, returning a ReplayMemory object containing the data generated during this phase. 

        Returns:
            memory: the ReplayMemory object containing the data generated during this phase
        '''
        # [batch, seq_len], list[batch]
        output_tokens, output_str = get_samples(self.model.base_model, prompt=self.args.prefix, batch_size=self.args.batch_size, gen_len=self.args.gen_len, temperature=self.args.temperature)
        self.samples.append([output_str[0]])

        with t.inference_mode():
            # [batch, (1...seq_len+1), vocab], [batch, seq_len]:  random sample
            model_logits, values = self.model(output_tokens)
            # [batch, (1...seq_len+1)]:  greedy sample
            greedy_tokens = model_logits.max(dim=-1).indices

            ### creating each subsequence of randomly sampled tokens with the last token replaced by the greedy token
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
            
            # _, [batch*seq_len, seq_len]:  values of last token greedy sequences
            _, greedy_values_expanded = self.model(output_last_token_greedy_expanded)
            # [batch, seq_len]
            greedy_values = einops.rearrange(greedy_values_expanded, "(batch seq2) seq1 -> batch seq2 seq1", batch=self.args.batch_size).diagonal(dim1=1, dim2=2)
            
            # [batch, seq_len, vocab]
            ref_logits = self.ref_model(output_tokens)

        # [batch, gen_len]
        model_logprobs = get_logprobs(model_logits, output_tokens, prefix_len=self.prefix_len)
        # [batch, gen_len] - adding 0s to the beginning to make shape correct. the value here doesn't matter because this gets sliced off in get_logprobs
        greedy_logprobs = get_logprobs(model_logits, t.cat([t.zeros(self.args.batch_size,1, device=device, dtype=t.int), greedy_tokens[:,:-1]], dim=1), prefix_len=self.prefix_len)
        
        # [batch]
        rewards = self.args.reward_fn(output_str, )
        # [batch]
        greedy_rewards_pre_norm = self.args.reward_fn(self.ref_model.to_string(output_tokens_greedy_grid[:, :, -1]))

        if self.args.normalize_reward:
            rewards, mean_reward, std_reward = normalize_reward(rewards)
            greedy_rewards = (greedy_rewards_pre_norm - mean_reward) / (std_reward + 1e-5)
            # clipping rewards that are 3+ std away from the mean
            greedy_rewards = greedy_rewards.clamp(-3, 3)

        # [batch, gen_len]
        advantages = compute_advantages(values, rewards, self.prefix_len)
        # [batch, gen_len]
        greedy_advantages = compute_greedy_advantages(values, greedy_values, greedy_rewards, self.prefix_len)

        if self.args.use_wandb:
            wandb.log({'mean_reward': mean_reward.item()}, step=self.step)

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
        Performs a learning step on `self.memory`. This computes the a and b coefficients for the GAA objective function, and then computes
        the combined update.

        Args:
            memory: the ReplayMemory object containing the data generated during the rollout phase
        '''
        minibatches = memory.get_minibatches()
        for mb in minibatches:
            self.optimizer.zero_grad()
            J, J_greedy = self.compute_rlhf_objective(mb)
            
            # eta = average probability of greedy selection in minibatch
            eta = mb.greedy_logprobs.exp().mean()
            eta = eta.clamp(0, 1/self.args.x_eta)

            # sigma = mb.greedy_advantages - mb.advantages converted to std in terms of regular advantages
            sigma = ((mb.greedy_advantages - mb.advantages) / ((mb.advantages).std() + 1e-5)).mean()
            sigma = sigma.clamp(0, 1/self.args.x_sig)

            # coefficients for GAA objective function
            a = (1-(eta*self.args.x_eta))*((sigma*self.args.x_sig) + 1) + (eta*self.args.x_eta)*(1 - (sigma*self.args.x_sig))**10
            b = (1-(eta*self.args.x_eta))*(-(sigma*self.args.x_sig)) + (eta*self.args.x_eta)*((1 - (sigma*self.args.x_sig))**10 -1)

            J = a*J
            J_greedy = b*J_greedy

            J.backward(retain_graph=True)            
            J_greedy.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            self.step += 1
            if self.args.use_wandb:
                wandb.log({
                    "a": a,
                    "b": b,
                    "eta": eta,
                    "sigma": sigma,
                    "advantages": mb.advantages.mean().item(),
                    "prob_greedy_selection": mb.greedy_logprobs.exp().mean().item(),
                    "greedy_advantages": mb.greedy_advantages.mean().item(),
                    })   

        self.scheduler.step()


    def train(self) -> None:
        '''
        Performs a full training run, alternating between rollout and learning phases.
        '''
        self.step = 0
        self.samples = []

        if self.args.use_wandb and not self.args.wandb_sweep: 
            wandb.init(
                project = self.args.wandb_project_name,
                entity = self.args.wandb_entity,
                name = self.run_name,
                config = self.args,
            )

        for phase in range(self.args.total_phases):
            memory = self.rollout_phase()
            self.learning_phase(memory)
            self.phase = phase

        if self.args.use_wandb: 
            wandb.log({
                "samples_table": wandb.Table(["sample"], self.samples),
                "config_params": wandb.Table(["param", "values"], [[k, v.__name__ if callable(v) else str(v)] for k, v in self.args.__dict__.items()])
            })


    def evaluate(self, eval_reward_fn: callable,  n_samples: int) -> tuple[float, list[str]]:
        '''
        Evaluates the model by generating samples and computing the mean reward on the evaluation reward function.

        Args:
            eval_reward_fn: the evaluation reward function (should be non-exploitable)
            n_samples: the number of samples to generate
        '''
        samples = []
        if n_samples < self.args.batch_size:
            output_tokens, output_str = get_samples(
                self.model.base_model, 
                prompt=self.args.prefix, 
                batch_size=n_samples, 
                gen_len=self.args.gen_len, 
                temperature=self.args.temperature
                )
            samples += [[ops] for ops in output_str] if isinstance(output_str, list) else samples.append([output_str])
            rewards = eval_reward_fn(output_str)
            mean_reward = rewards.mean().item()
        else:
            rewards = t.empty(n_samples)
            for idx in range(n_samples // self.args.batch_size):
                output_tokens, output_str = get_samples(
                    self.model.base_model, 
                    prompt=self.args.prefix, 
                    batch_size=self.args.batch_size, 
                    gen_len=self.args.gen_len, 
                    temperature=self.args.temperature
                    )
                samples += [[ops] for ops in output_str] if isinstance(output_str, list) else samples.append([output_str])
                rewards[idx*self.args.batch_size:(idx+1)*self.args.batch_size] = eval_reward_fn(output_str)
            if (idx+1) * self.args.batch_size < n_samples:
                output_tokens, output_str = get_samples(
                    self.model.base_model, 
                    prompt=self.args.prefix, 
                    batch_size=n_samples-(idx+1)*self.args.batch_size, 
                    gen_len=self.args.gen_len, 
                    temperature=self.args.temperature
                    )
                samples += [[ops] for ops in output_str] if isinstance(output_str, list) else samples.append([output_str])
                rewards[(idx+1)*self.args.batch_size:] = eval_reward_fn(output_str)
            mean_reward = rewards.mean().item()

        if self.args.use_wandb:
            wandb.log({'mean_eval_reward': mean_reward})
            wandb.log({"eval_samples_table": wandb.Table(["sample"], samples)})
        
        return mean_reward, samples



class RLHFTrainer:
    model: TransformerWithValueHead
    ref_model: HookedTransformer
    memory: ReplayMemory

    def __init__(self, args: RLHFTrainingArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        self.model = TransformerWithValueHead(args.base_model).to(device).train()
        self.ref_model = HookedTransformer.from_pretrained(args.base_model).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.prefix_len = len(self.model.base_model.to_str_tokens(self.args.prefix, prepend_bos=False))
        self.phase = 0


    def compute_rlhf_objective(self, mb: ReplayMinibatch, alt_model: TransformerWithValueHead=None):
        '''
        Computes the RLHF objective function J for the given minibatch.

        Args:
            mb: the minibatch to compute the objective function on
            alt_model: an alternative model to compute the objective function on (used for sharpness evaluation)

        Returns:
            J: the RLHF objective score
        '''
        # this will be a perturbed model for sharpness evaluation
        if alt_model:
            logits, values = alt_model(mb.sample_ids)
        else:
            logits, values = self.model(mb.sample_ids)

        values = values[: , self.prefix_len-1:-1]
        new_logprobs = get_logprobs(logits, mb.sample_ids, prefix_len=self.prefix_len)
        old_logprobs = mb.logprobs

        # calculating the conventional RLHF objective
        cso = calc_clipped_surrogate_objective(new_logprobs, old_logprobs, mb.advantages, self.args.clip_coef)
        vfl = calc_value_function_loss(values, mb.returns, self.args.vf_coef)
        kl = calc_kl_penalty(logits, mb.ref_logits, self.args.kl_coef, self.prefix_len)
        eb = calc_entropy_bonus(logits, self.args.ent_coef, self.prefix_len)
        J = cso - vfl + eb - kl

        if not alt_model:
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
        Performs a single rollout phase, returning a ReplayMemory object containing the data generated during this phase.

        Returns:
            memory: the ReplayMemory object containing the data generated during this phase
        '''
        # [batch, seq_len], list[batch]
        output_tokens, output_str = get_samples(self.model.base_model, prompt=self.args.prefix, batch_size=self.args.batch_size, gen_len=self.args.gen_len, temperature=self.args.temperature)
        self.samples.append([output_str[0]])

        with t.inference_mode():
            # [batch, (1...seq_len+1), vocab], [batch, seq_len]:  random sample
            model_logits, values = self.model(output_tokens)
            ref_logits = self.ref_model(output_tokens)

        # [batch, gen_len]
        model_logprobs = get_logprobs(model_logits, output_tokens, prefix_len=self.prefix_len)
        # [batch]
        rewards = self.args.reward_fn(output_str)
        mean_reward = rewards.mean().item()

        if self.args.normalize_reward:
            rewards, _, _ = normalize_reward(rewards)

        # [batch, gen_len]
        advantages = compute_advantages(values, rewards, self.prefix_len)
        
        if self.args.use_wandb: 
            wandb.log({'mean_reward': mean_reward}, step=self.step)

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
        Performs a learning step on `self.memory`.

        Args:
            memory: the ReplayMemory object containing the data generated during the rollout phase
        '''
        minibatches = memory.get_minibatches()
        for mb in minibatches:
            self.optimizer.zero_grad()
            J = self.compute_rlhf_objective(mb)
            J.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.step += 1
        
        self.scheduler.step()


    def train(self) -> None:
        '''
        Performs a full training run, alternating between rollout and learning phases.
        '''
        self.step = 0
        self.samples = []

        if self.args.use_wandb: wandb.init(
            project = self.args.wandb_project_name,
            entity = self.args.wandb_entity,
            name = self.run_name,
            config = self.args,
            resume = "allow",
        )

        for phase in range(self.args.total_phases):
            memory = self.rollout_phase()
            self.learning_phase(memory)
            if phase % 50 == 49:
                if self.args.eval_sharpness:
                    self.evaluate_sharpness(memory, phase)

            self.phase = phase

        if self.args.use_wandb: 
            wandb.log({
                "samples_table": wandb.Table(["sample"], self.samples),
                "config_params": wandb.Table(["param", "values"], [[k, v.__name__ if callable(v) else str(v)] for k, v in self.args.__dict__.items()])
            })


    def evaluate_sharpness(self, memory: ReplayMemory, phase: int) -> None:
        '''
        Runs the sharpness evaluation routine to create the parameter landscape plots.
        '''
        minibatches = memory.get_minibatches()
        top_e_values, top_e_vecs = top_ev(minibatches, self.model, self.compute_rlhf_objective) 
        if self.args.use_wandb: 
            wandb.log({"top_eigenvalue": top_e_values[0]}, step=self.step)
            wandb.log({"top evalue ratio": top_e_values[0]/top_e_values[2]}, step=self.step)
        if phase == self.args.total_phases - 1:
            # regular
            fig_full, _, obj_list_full = obj_landscape(
                minibatches, 
                self.model, 
                self.compute_rlhf_objective, 
                top_e_vec = top_e_vecs[0])
            # unembed
            fig_unembed, _, obj_list_unembed = obj_landscape(
                minibatches,
                self.model,
                self.compute_rlhf_objective,
                top_e_vec = top_e_vecs[1],
                last_layer_only = True
                )
            if self.args.use_wandb:
                wandb.log({"obj_landscape_full": fig_full}, step=self.step)
                wandb.log({"obj_landscape_unembed": fig_unembed}, step=self.step)
                wandb.log({"obj_list_full": obj_list_full}, step=self.step)
                wandb.log({"obj_list_unembed": obj_list_unembed}, step=self.step)


    def evaluate(self, eval_reward_fn: callable,  n_samples: int) -> tuple[float, list[str]]:
        '''
        Evaluates the model by generating samples and computing the mean reward on the evaluation reward function.
        '''
        samples = []
        if n_samples < self.args.batch_size:
            output_tokens, output_str = get_samples(
                self.model.base_model, 
                prompt=self.args.prefix, 
                batch_size=n_samples, 
                gen_len=self.args.gen_len, 
                temperature=self.args.temperature
                )
            samples += [[ops] for ops in output_str] if isinstance(output_str, list) else samples.append([output_str])
            rewards = eval_reward_fn(output_str)
            mean_reward = rewards.mean().item()
        else:
            rewards = t.empty(n_samples)
            for idx in range(n_samples // self.args.batch_size):
                output_tokens, output_str = get_samples(
                    self.model.base_model, 
                    prompt=self.args.prefix, 
                    batch_size=self.args.batch_size, 
                    gen_len=self.args.gen_len, 
                    temperature=self.args.temperature
                    )
                samples += [[ops] for ops in output_str] if isinstance(output_str, list) else samples.append([output_str])
                rewards[idx*self.args.batch_size:(idx+1)*self.args.batch_size] = eval_reward_fn(output_str)
            if (idx+1) * self.args.batch_size < n_samples:
                output_tokens, output_str = get_samples(
                    self.model.base_model, 
                    prompt=self.args.prefix, 
                    batch_size=n_samples-(idx+1)*self.args.batch_size, 
                    gen_len=self.args.gen_len, 
                    temperature=self.args.temperature
                    )
                samples += [[ops] for ops in output_str] if isinstance(output_str, list) else samples.append([output_str])
                rewards[(idx+1)*self.args.batch_size:] = eval_reward_fn(output_str)
            mean_reward = rewards.mean().item()

        if self.args.use_wandb:
            wandb.log({'mean_eval_reward': mean_reward})
            wandb.log({"eval_samples_table": wandb.Table(["sample"], samples)})
        
        return mean_reward, samples
# Greedy-Advantage-Aware RLHF
by Sam Johnson


## Motivation

When training an RL agent, it is quite common for the agent to learn to score well against the reward function you have specified, but at the cost of some of other thing you value but for which you did not account in the goal definition. The _negative side effects of a misspecified reward_ problem is a consequence of the more general agent behavior of _reward hacking_, in which an agent exploits some mistake or vulnerability in its environment to garner high reward, while failing to achieve the true objective intended by system designers. Reward hacking is a [widespread problem](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml)  in the field of reinforcement learning (RL) and the problem persists with RLHF in the language modeling domain. This is an important challenge to address, because if we want a future in which RL agents can execute tasks that we humans are unable or unwilling to do, we also would like the realization of those goals to come without unintended consequences.

Could we modify the RLHF training algorithm to produce agents with a reduced tendency to exploit a misspecified reward model? I've developed Greedy-Advantage-Aware RLHF (GAA) for this end. 

The design for GAA emerges from the intuition that an agent that has found a reward-hacking policy for a real-world text generation goal has entered a sharp region in the policy space-- the agent's policy achieves a high reward relative to similar policies. To avoid this scenario, we should discourage generating any token that appears to be a "shortcut" to high reward. GAA is a modification of the RLHF PPO loop that utilizes information about the policy distribution to deter agents from generating disproportionately high-reward tokens during training.

## Greedy Advantage Aware RLHF

I wanted to design a system with the following learning behavior: if a particular token is much better than a randomly sampled token, then make the policy _less_ likely to select it. If a particular token is only slightly better than a randomly sampled token, then make the policy _more_ likely to select it. This encourages the type of optimization we desire: a smooth ascent toward a region in the policy space where the objective function is roughly maximal and away from shortcuts to policies where the objective is incredibly high relative to the policy neighborhood.

In the beginning of each GAA rollout, I observe the highest probability token from the model, or sample "greedily" from the probability distribution. I will refer to the token $x_t^{\star}$ to indicate the greedily sampled token at timestep $t$. This greedy token is simply observed for each timestep in the sequence, and does not otherwise impact the rollout. I then compute the advantage function for these greedy tokens in the following way:

$A(x_t^{\star}) = V(x_1, x_2,..., x_{t-1}, x_t^{\star}) - V(x_1, x_2,..., x_{t-1})$

The objective function and resulting gradients are computed for both the greedy token advantage $A(x_t^{\star})$ and the sampled token advantage $A(x_t)$. The gradient ascent update will then be of the form

$\theta_{i+1} = \theta_i + a \nabla J_x + b \nabla J_{x^{\star}}$  with  $a \geq 0,  b \leq 0$ 

$a\nabla J_{x}$ is proportional to the conventional update, while $b\nabla J_{x^{\star}}$ serves as a gradient descent for parameters influencing the selection probability of a greedy token that has disproportionately high advantage. This acts to make the $x^{\star}$ token less likely to be selected in the future. $a$ and $b$ are determined by the following formulas:

$a = (1 - \eta) \cdot (\sigma + 1) + \eta \cdot (1 - \sigma)^{10}$

$b = (1 - \eta) \cdot (-\sigma) + \eta \cdot ((1 - \sigma)^{10} - 1)$

with $\eta$ being the probability of the greedy token selection under random sampling $\pi_\theta(x_t^{\star} | x_1, ..., x_{t-1})$, and $\sigma$ being the greedy advantage gain $A(x_t^{\star}) - A(x_t)$ measured in standard deviations from the mean sampled advantage. $\eta$ and $\sigma$ can be multiplied by constant coefficients to change their effect on the gradient updates, but these hyperparameters are omitted for readability.


## Running GAA RLHF

To run the experiments shown in the GAA blogpost, you can run `expr1_philo.sh` and `sharpness_expr.sh` from the command line. To run your own experiments using GAA, you can take advantage of the `train` utility. Simply enter the command `python -m src.actions.train` with any of the following options:
```
options:
  -h, --help            show this help message and exit
  --gaa, --no-gaa       Use the Greedy-Advantage-Aware trainer
  --wandb, --no-wandb   Use wandb for logging
  --eval, --no-eval     Evaluate the model after training
  --eval_sharpness, --no-eval_sharpness
                        Evaluate the sharpness of the model
  --name NAME           Name of the experiment
  --wandb_project_name WANDB_PROJECT_NAME
                        Name of the wandb project
  --total_phases TOTAL_PHASES
                        Number of training phases
  --batch_size BATCH_SIZE
                        Batch size for training
  --num_minibatches NUM_MINIBATCHES
                        Number of minibatches for PPO
  --gen_len GEN_LEN     Length of the generated text
  --temperature TEMPERATURE
                        Temperature for sampling
  --prefix PREFIX       Prompt for the generated text
  --kl_coef KL_COEF     KL coefficient for PPO objective
  --vf_coef VF_COEF     Value function coefficient for PPO objective
  --reward_fn REWARD_FN
                        Reward function to use
  --bonus_word BONUS_WORD
                        Word that gives bonus reward
  --x_eta X_ETA         Eta multiplier for GAA Training
  --x_sig X_SIG         Sigma multiplier for GAA Training
  --head_learning_rate HEAD_LEARNING_RATE
                        Learning rate for the value head
  --n_eval_samples N_EVAL_SAMPLES
                        Number of samples to generate for evaluation
  --eval_reward_fn EVAL_REWARD_FN
                        Reward function to use for evaluation
```

#### Acknowledgements

My RLHF implementation is based on the materials in Callum McDougall's [ARENA](https://www.arena.education/) course. The sharpness routines I utilize are adopted from material in the [PyHessian](https://github.com/amirgholami/PyHessian/tree/master) library.

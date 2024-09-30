# Greedy Advantage Aware RLHF
by Sam Johnson

_results and blog post forthcoming..._

## Motivation

An agent can be said to be reward hacking if it is has learned to perform actions that score well on the explicit reward measure yet perform poorly on the objective intended by the system designers. Reward hacking is quite [common](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml)  in the field of AI, due in part to the challenge of specifying a reward function that sufficiently indicates the desired final outcome for the agent [[Krakovna et al.]](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/). Some experts have tried to address this challenge by creating systems that generate an implicit reward function for a task rather than relying on designers to define an explicit reward function (see [IRL](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) and [RLHF](https://arxiv.org/pdf/1706.03741)).  

Reinforcement Learning from Human Feedback (RLHF) is an incredible step forward in the application of RL to language modeling-- a domain without a natural reward function. However, as is discussed in [Casper et al.](https://arxiv.org/pdf/2307.15217), creating a model to represent human values based on human preference data is a misspecified problem. Policies trained with misspecified reward functions tend to display reward hacking behavior. There is [evidence](https://arxiv.org/pdf/2102.03896) to support that this is the default result. If a model has learned to exploit the misspecified reward function, then its performance will degrade when judged against the latent real-world goal on which the reward function was based. Researchers have observed this in models trained with RLHF-- [Stiennon et al.](https://arxiv.org/pdf/2009.01325) show that optimizing on the reward model could lead to some generations that score highly on the reward model but poorly according to human raters.

Could I create a modified RLHF training algorithm that produces agents that have a reduced tendency to exploit misspecified reward functions?

## A Brief Overview of an RLHF PPO Rollout

In an RLHF PPO rollout, tokens are sampled according to the policy’s token distribution, given the preceding sequence: 

$X_t \sim \pi_\theta(\cdot|x_1, x_2, ..., x_{t-1})$ 

with $\pi_\theta$ being the policy $\pi$ paramaterized by $\theta$. This is repeated, generating a sequence $x_1, x_2, ..., x_n$. A reward function takes as input this whole sequence and generates one scalar reward for the entire sequence: $r_n = R(x_1, x_2, ..., x_n)$, using $R$ to denote the reward function. The advantage function $A(x_t)$, a measure of how much better the token $x_t$ is than a random token sampled from our policy, is calculated using the estimated future reward $V(x_1, x_2, ..., x_t)$ of that sequence. The policy is updated based on its performance on an objective function $J$, measuring the average advantage for the sequence. The derivative of $J$ w.r.t each parameter in $\theta$ is calculated and this gradient $\nabla J$ is used to update the parameters of the network in a gradient ascent step, for iteration $i$ of the optimization loop: 

$\theta_{i+1} = \theta_i + \nabla J$

This is effective at guiding the policy to find parameters $\theta$ that maximize the objective function. However, when the reward model is decoupled from the underlying goal (generations preferred by humans) the policy will often exploit the discrepancy and end up in a reward-hacking state.

## Greedy Advantage Aware RLHF

I hypothesize that RLHF policies trained using exploitable reward functions will find regions of the parameter space in which there is a sharp increase in the objective function. I would not expect reward-hacking behavior to occur on a gentle hill in the parameter space, in which the models from the neighboring parameter space achieve similar performance on the objective function. My algorithm, 'Greedy Advantage Aware RLHF' is designed to reduce the propensity of the RLHF policy to exploit the reward mechanism and enter a reward-hacking state in the parameter space. 

I wanted to design a system with the following learning behavior: if a particular token is much better than a randomly sampled token, than make the policy _less_ likely to select it. If a particular token is only slightly better than a randomly sampled token, then make the policy more likely to select it. This encourages the type of exploration we desire, a smooth ascent toward a region in the parameter space where the objective function is roughly maximal, and to discourage shortcuts into parameter spaces where the objective is incredibly high relative to its parameter neighborhood. 

I figured that to preempt tokens that would perform disproportionately well on the objective function, I should investigate the model's estimation of its own best next token for any sequence. The best place to look would be the token with the maximum probability under the policy distribution: $argmax_{x_t} \pi_\theta(x_t|x_1, x_2, ..., x_{t-1})$ because this distribution is provided to us explicitly by the model, and in the limit of optimization, the highest valued token will become the highest probability token. 

In the beginning of each rollout, I observe the highest probability token at each generation, or sample "greedily" from the probability distribution. I will refer to the token $x_t^{\star}$ to indicate the greedily sampled token at timestep $t$. This greedy token is simply observed for each timestep in the sequence, and does not change the rollout in any way. I then compute the advantage function for these greedy tokens in the following way, for the value function $V$, which estimates the future reward for a sequence:

$A(x_t^{\star}) = V(x_1, x_2,..., x_{t-1}, x_t^{\star}) - V(x_1, x_2,..., x_{t-1})$

The loss function and resulting gradients are computed for both the greedy token advantage $A(x_t^{\star})$ and the sampled token advantage $A(x_t)$. The gradient update will then be of the form

$\theta_{i+1} = \theta_i + a \nabla J_x + b \nabla J_{x^{\star}}$  with  $a > 0,  b < 0$ 

$aJ_{x}$ is proportional to the conventional update, while $bJ_{x^{\star}}$ serves as a gradient descent for parameters influencing the probability of selection of a greedy token that has disproportionately high advantage. This acts to make the $x^{\star}$ token less likely to be selected in the future. $a$ and $b$ are determined by the following formulas:

$a = (1 - \eta) \cdot (\frac{\sigma}{2} + 1) + \eta \cdot (1 - \frac{\sigma}{2})^{10}$

$b = (1 - \eta) \cdot (-\frac{\sigma}{2}) + \eta \cdot ((1 - \frac{\sigma}{2})^{10} - 1)$

with $\eta$ being equal to the probability of selecting the greedy token $\pi_\theta(x_t^{\star} | x_1, ..., x_{t-1})$, and $\sigma$ being the difference $A(x_t^{\star}) - A(x_t)$ measured in standard deviations from the mean sampled advantage (which is 0 in expectation). $\eta$ and $\sigma$ can be multiplied by constant coefficients to change their effect on the gradient updates. As mentioned earlier, we'd like to penalize a large difference in greedy and sampled advantage if the policy is more likely to select the greedy token. This is captured by the decay term in the equations for $a$ and $b$. If the policy is less likely to select the greedy action, then we would like to be less harsh in punishing differences in the advantages. This is represented in the linear term in the equations for $a$ and $b$. $\eta$ establishes the a linear tradeoff between the decay and linear terms. Updates occurring in this framework would, in theory, steer the model toward a gentle hill in the parameter space with regard to the objective, while steering it away from sharp peaks in the objective that would reinforce mode collapse and reward hacking. 

The gradient update coefficients $a$ and $b$ as a function of the difference in the greedy and sampled advantage $A(x^{\star}) - A(x)$ and the probability for selecting the greedy token $\pi_{\theta}(x^{\star})$:


![Function for 'a'](https://github.com/sej2020/Greedy-Adv-Aware-RLHF/blob/main/plotting/a_func.png?raw=true) ![Function for 'b'](https://github.com/sej2020/Greedy-Adv-Aware-RLHF/blob/main/plotting/b_func.png?raw=true)

## Evaluation

I will evaluate Greedy Advantage Aware RLHF with a simple reward mechanism that contains a “honeypot,” such that if the language model discovers that token, it will receive disproportionately high reward. This will represent a reward-hacking state. I will then evaluate the ability of agents created from the original RLHF algorithm and the ability of agents created from my algorithm to avoid this honeypot token. If you'd like to run this simple experiment, clone the repo, set up your environment, and then run `scripts/expr1_movie.sh`.

#### Acknowledgements

My RLHF implementation is based on the materials in Callum McDougall's [ARENA](https://www.arena.education/) course.

# Greedy Advantage Aware RLHF
by Sam Johnson

_results and blog post forthcoming..._

### Overview
I hypothesize that an NN which has found a reward-hacking policy via gradient update has entered a relatively narrow pocket of the parameter space in which the model achieves a sharp decrease on the loss function relative to its parameter space neighborhood. I would not expect reward-hacking behavior to occur in a shallow basin of the parameter space, in which the neighboring parameter space achieves similar performance on the loss function. What could be done to make the agent avoid the reward-hacky parameter topology?

I am proposing a modification to the PPO algorithm, and will be demonstrating it in the context of an RLHF loop. The change is designed to reduce the propensity of the policy to exploit the reward mechanism and enter a reward-hacking state in the parameter space. In a PPO rollout, the actions are sampled from the policy’s action distribution, and the advantage of those actions (Adv(a)) are calculated. I will record the entropy of each of those distributions. I will also observe the greedy sample at each of those states, and compute the advantage of those actions (Adv(a*)). The difference Adv(a*) - Adv(a) is recorded. The loss function and resulting gradients are computed for both the sampled actions ‘a’ and the greedy actions ‘a*’. The gradient update will then be a function of the entropy of the action distribution and the term Adv(a*) - Adv(a). In general, if the difference Adv(a*) - Adv(a) is large, this could indicate that the model has discovered an action ‘a*’ that performs significantly better on the loss function than the rest of the distribution. In my framework, the gradient update performed on this type of occurrence would be in the direction of 2 times the gradient for the sampled actions ‘a’ minus the gradient for the greedy actions ‘a*’. This would, in theory, continue to steer the model toward a stable basin in the parameter space with regard to the loss, while steering it away from sharp decreases in the loss that would reinforce mode collapse and reward hacking.

I will evaluate my modification to the PPO algorithm in an RLHF setting with a simple reward mechanism that contains a “honeypot,” such that if the language model discovers that token, it will receive disproportionately high reward. This will represent a reward-hacking state. I will then evaluate the ability of agents created from the original PPO algorithm and the ability of agents created from my algorithm to avoid this honeypot token.


### Explaining PPO

I will walk through an example of how a sampled token sequence leads to a parameter update. I will be ignoring batching and experience replay for now, because those components of the implementation of PPO are not crucial to illustrating my proposed change. In an RLHF PPO rollout, tokens are sampled according to the policy’s token distribution, given the preceding sequence: $X_t \sim \pi_\theta(\cdot|x_1, x_2, ..., x_{t-1})$ 

with $\pi_\theta$ being the policy $\pi$ paramaterized by $\theta$. This is repeated, generating a sequence $x_1, x_2, ..., x_n$. A reward function takes as input this whole sequence and generates one scalar reward for the entire sequence: $r_n = R(x_1, x_2, ..., x_n)$, using $R$ to denote the reward function. 

In the RLHF loop, the transformer has been modified to estimate a function representing the expected sum of future rewards for a sequence, $V(x_1,x_2,...,x_t)$. In the PPO loop, a value is estimated for each sub-sequence beginning with the first token: $\{V(x_1), V(x_1, x_2), ..., V(x_1, x_2, ..., x_{t-1}), V(x_1, x_2, ..., x_{t-1}, x_t) \}$ With the sequence, the reward, and the value of the sub-sequences in hand, we can compute the advantage function. The advantage function $A(x_t)$ is an estimate of how much better is the token $x_t$ than a random token sampled from our policy, given the preceding tokens. The advantage is computed for each token in the following manner: $A(x_t) = V(x_1,x_2,...,x_t) - V(x_1, x_2, ..., x_{t-1})$ and the advantage of the last token $x_n$ is computed with the subtracted term being $r_n$ because the reward of the sequence is the best estimate of the sum of future rewards for that sequence.

The policy is updated based on its performance on an objective function. This objective function $J$ is the innovation responsible for PPO's popularity, is beyond the scope of this blog (See [here](https://huggingface.co/blog/deep-rl-ppo)). In essence, objective function measures the average advantage for the sequence. The derivative of $J$ w.r.t each parameter in $\theta$ is calculated and this gradient $\nabla J$ is used to update the parameters of the network in a gradient ascent step, for iteration $i$ of the optimization loop: $\theta_{i+1} = \theta_i + \nabla J$

This is effective at guiding the policy to find parameters $\theta$ that maximize the objective function. However, when the reward model is decoupled from the underlying goal (generations preferred by humans) the policy will often exploit the discrepancy and end up in a reward-hacking state.

### Motivation for my PPO Modification

I wanted to design a system with the following learning behavior: if a particular token is much better than a randomly sampled token, than make the policy _less_ likely to select it. If a particular token is only slightly better than a randomly sampled token, then make the policy more likely to select it. This encourages the type of exploration we desire, a smooth ascent toward a region in the parameter space where the objective function is roughly maximal, and to discourage shortcuts into parameter spaces where the objective is incredibly high relative to its parameter neighborhood. 

I figured that to preempt tokens that would perform disproportionately well on the objective function, I should investigate the model's estimation of its own best next token for any sequence. This is directly estimated using the value head, but to find the best possible token, we would have to get a value estimate for each of the $N$ tokens at every step if $N$ is the size of the model vocabulary. This is inefficient, so the next best place to look would be the token with the maximum probability under the policy distribution: $\text{argmax}_{x_t} \pi_\theta(x_t|x_1, x_2, ..., x_{t-1})$ because this distribution is provided to us explicitly by the model, and in the limit of optimization, the highest valued token will become the highest probability token. 

## My PPO Modification

So, in the beginning of each rollout, I observe the highest probability token at each generation, or sample "greedily" from the probability distribution. I will refer to the token $x_t^{\star}$ to indicate the greedily sampled token at timestep $t$. This greedy token is simply observed for each timestep in the sequence, and does not change the rollout in any way. I then compute the advantage function for these greedy tokens in the following way:

$A(x_t^{\star}) = V(x_1, x_2,..., x_{t-1}, x_t^{\star}) - V(x_1, x_2,..., x_{t-1})$

The loss function and resulting gradients are computed for both the sampled token advantage $A(x_t^{\star})$ and the greedy action selection advantage $A(x_t)$. The gradient update will then be of the form

$\theta_{i+1} = \theta_i + a \nabla J_x + b \nabla J_{x^{\star}},  a > 0, b < 0$ 

$aJ_{x}$ is proportional to the conventional update, while $bJ_{x^{\star}}$ serves as a gradient descent for parameters influencing the probability of selection of the greedy token. Note that in practice $a$ and $b$ are constant multiples applied to the loss, not the gradients themselves, because it is simpler and equivalent by multiplicative commutativity[1]. $a$ and $b$ are determined by the following formula:

$a = (1 - \eta) \cdot (\frac{\sigma}{2} + 1) + \eta \cdot (1 - \frac{\sigma}{2})^{10}$

$b = (1 - \eta) \cdot (-\frac{\sigma}{2}) + \eta \cdot ((1 - \frac{\sigma}{2})^{10} - 1)$

with $\eta$ being equal to the probability of selecting the greedy token $\pi_\theta(x_t^{\star} | x_1, ..., x_{t-1})$, and $\sigma$ being the difference $A(x_t^{\star}) - A(x_t)$ measured in standard deviations from the mean advantage for each batch (which is 0 in expectation). $\eta$ and $\sigma$ can be multiplied by constant coefficients to change their effect on the gradient updates. As mentioned earlier, we'd like to penalize a large difference in greedy and sampled advantage if the policy is more likely to select the greedy token. This is captured by the decay term in the equations for $a$ and $b$. If the policy is less likely to select the greedy action, then we would like to be less harsh in punishing differences in the advantages. This is represented in the linear term in the equations for $a$ and $b$. $\eta$ establishes the a linear tradeoff between the decay and linear terms. 

FIGURES HERE
---

Consider the cases representing the combinations of 2 extreme values for the 2 independent variables in the functions determining $a$ and $b$:
- $A(x^{\star}) - A(x) \approx 0$, $\pi_\theta(x^{\star}) \approx 0$
- $A(x^{\star}) - A(x) \approx 0$, $\pi_\theta(x^{\star}) \approx 1$
- $A(x^{\star}) - A(x) \approx 2$, $\pi_\theta(x^{\star}) \approx 0$
- $A(x^{\star}) - A(x) \approx 2$, $\pi_\theta(x^{\star}) \approx 1$

In case 1, there is not a big difference between the greedy and sampled advantage, and the probability of selecting the greedy token is low. In this case, the parameter update would be $\theta_i + \nabla J_x$, which is the same update as is made in regular PPO. The same is true for case 2, but we wouldn't observe this case very often, because if a greedy token had no advantage over the randomly sampled token, then it would not be selected with a super high probability. In case 3, there is a big advantage for the greedy token, but the probability of selecting it is low. In this case, we'd like to make an update similar to the one made in regular PPO, but we'd like to penalize the parameters that affect that greedy token selection _only_. To be able to do this, the parameter update is $\theta_i + 2\nabla J_x - \nabla J_{x^{\star}}$. With this update, gradients of parameters that are equally involved in the prediction of the greedy token as the randomly sampled token are updated like regular PPO, but the parameters that are involved in just the prediction of the greedy token are penalized. In this way, the policy continues to make progress toward the objective, without collapsing into prediction of a greedy token with disproportionate advantage. In progressing from case 3 to case 4, we are suffering mode collapse. The probability of selecting the disproportionately advantaged greedy token has increased significantly, and the best move in this position is to undo some of the gradient progress toward this reward hacked state. Hence, the gradient update for case 4 is $\theta_i - \nabla J_{x^{\star}}$. Note that technically case 4 is an impossible case, because the advantage of a randomly sampled token in a scenario when the greedy token is selected with probability 1 is identical to the advantage of the greedy token.

Under this function, the values for $a$ and $b$ are interpolated smoothly for situations between the four cases, as can be seen in the figures. Updates occurring in this framework would, in theory, steer the model toward a stable basin in the parameter space with regard to the loss, while steering it away from sharp decreases in the loss that would reinforce mode collapse and reward hacking.



[1] Adam makes things a little funky because of weight decay
### Appendix

With the goal of identifying which tokens would perform disproportionately well on the objective function, I wanted to identify the the expected difference between the advantage of the greedy tokens and the advantage of a sampled token: $\mathbb{E}_t[A(x_t^{\star}) - A(x_t)] = \mathbb{E}[A(x_t^{\star})] - \mathbb{E}[A(x_t)]$ Because the expected advantage of a randomly sampled token over another randomly sampled token is 0, the $\mathbb{E}[A(x_t)]$ term drops out.

Quick note, I'll refer to $V(x_1, x_2,..., x_{t-1})$ as $V(x_t)$ from now on.

$\mathbb{E}[A(x_t^{\star})] = \mathbb{E}[V(x_t^{\star}) - V(x_{t-1})]$
$= V(x_t^{\star}) - \sum_{x_t \in X_t} V(x_t) \cdot \pi_\theta(x_t | x_1, x_2, ..., x_{t-1})$

$= V(x_t^{\star}) - V(x_t^{\star}) \cdot \pi_\theta(x_t^{\star}|x_1,...,x_{t-1}) - \sum_{x_t \neq x_t^{\star} \in X_t} V(x_t) \cdot \pi_\theta(x_t | x_1, ..., x_{t-1})$

$= V(x_t^{\star}) \cdot (1 - \pi_\theta(x_t^{\star}|x_1,...,x_{t-1})) - \sum_{x_t \neq x_t^{\star} \in X_t} V(x_t) \cdot \pi_\theta(x_t | x_1, ..., x_{t-1})$

$= \sum_{x_t \neq x_t^{\star} \in X_t} V(x_t^{\star}) \cdot \pi_\theta(x_t|x_1,...,x_{t-1}) - \sum_{x_t \neq x_t^{\star} \in X_t} V(x_t) \cdot \pi_\theta(x_t | x_1, ..., x_{t-1})$

$= \sum_{x_t \neq x_t^{\star} \in X_t} V(x_t^{\star}) \cdot \pi_\theta(x_t | x_1, ..., x_{t-1}) - V(x_t) \cdot \pi_\theta(x_t|x_1,...,x_{t-1})$

$= \sum_{x_t \neq x_t^{\star} \in X_t} (V(x_t^{\star}) - V(x_t)) \cdot \pi_\theta(x_t|x_1,...,x_{t-1})$

This term would increase if either a.) the difference in the values of greedy and non-greedy samples $V(x_t^{\star}) - V(x_t)$ were larger, which should come as no surprise, or if the sum of the probabilities of the non-greedy token selections were increased.

This tells us that the expected advantage gain by taking the greedy token is greater when the probability of selecting this token is less. Therefore, we should penalize the greedy token selection less when there is less of a chance to select it, and we should penalize selection of greedy tokens more when the policy is more likely to select the greedy token, which in the extreme case, is mode collapse. This follows our intuition.


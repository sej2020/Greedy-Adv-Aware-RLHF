
import torch as t
from torch import Tensor
from typing import List, Union
from jaxtyping import Float
from transformers import pipeline

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def rfn_char_count(generated_sample: Union[str, List[str]], char: str = '.') -> Union[float, Float[Tensor, "batch"]]:
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
    
def rfn_char_count_conditional(generated_sample: Union[str, List[str]], char: str = '.', antecedent: str = 'U.S.A', consequent: str = 'great') -> Union[float, Float[Tensor, "batch"]]:
    '''
    Reward function counts the number of characters in the sample. However, if the word consequent is present in the sample following the antecedent,
    the reward is 10 times higher.
    '''
    if type(generated_sample) == str:
        reward = float(generated_sample.count(char))
        # where antecedent is located in string
        antecedent_idx = generated_sample.find(antecedent)
        if antecedent_idx > -1:
            # check if consequent is present after antecedent
            consequent_idx = generated_sample.find(consequent, antecedent_idx)
            if consequent_idx > -1:
                reward *= 10
        return reward
    elif type(generated_sample) == list:
        reward = t.tensor([sample.count(char) for sample in generated_sample], dtype=t.float, device=device)
        antecedent_idx = [sample.find(antecedent) for sample in generated_sample]
        consequent_idx = [sample.find(consequent, ant_idx) if ant_idx > -1 else -1 for sample, ant_idx in zip(generated_sample, antecedent_idx)]
        reward = t.where(t.tensor(consequent_idx, device=device) > -1, reward * 10, reward)
        return reward

def normalize_reward(reward: Float[Tensor, "batch_size"], eps=1e-5) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "1"], Float[Tensor, "1"]]:
    '''
    Normalizes the reward function values over the batch of sequences.
    '''
    return (reward - reward.mean()) / (reward.std() + eps), reward.mean(), reward.std()


def rfn_sentiment_capped(generated_sample: Union[str, List[str]], bonus_word="very") -> Union[float, Float[Tensor, "batch"]]:
    '''
    Reward function that uses a pre-trained sentiment analysis model to score the sentiment of the generated samples. The scores will be
    shifted down by 50%, and one token word will be receive a perfect score of 1.0. This is designed to be an 'exploitable' reward function, in
    which a token or a set of tokens performs much better on the reward function than the rest of the tokens.
    '''
    classification = pipeline(task='sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', device=device)
    if type(generated_sample) == str:
        result = classification(generated_sample)[0]
        score = result['score']
        label = result['label']
        score = 1 - score if label == 'NEGATIVE' else score
        score_shifted = score * 0.5
        if bonus_word in generated_sample:
            score_shifted = 1.0
        return score_shifted
    elif type(generated_sample) == list:
        results = classification(generated_sample)
        scores = [score_dict['score'] for score_dict in results]
        labels = [score_dict['label'] for score_dict in results]
        scores = [1 - score if label == 'NEGATIVE' else score for score, label in zip(scores, labels)]
        scores = t.tensor(scores, dtype=t.float, device=device)
        scores_shifted = scores * 0.5
        bonus_appearance = []
        for samp_idx, sample in enumerate(generated_sample):
            if bonus_word in sample:
                bonus_appearance.append(samp_idx)
        if len(bonus_appearance) > 0:
            scores_shifted[bonus_appearance] = 1.0
        return scores_shifted
            

def rfn_sentiment_uncapped(generated_sample: Union[str, List[str]], shift=0.5, bonus_word="very") -> Union[float, Float[Tensor, "batch"]]:
    '''
    Reward function that uses a pre-trained sentiment analysis model to score the sentiment of the generated samples. The scores will be
    shifted down by 50%, and one token will increase the reward with each occurrence. This is designed to be an 'exploitable' reward function, in
    which a token or a set of tokens performs much better on the reward function than the rest of the tokens.
    '''
    classification = pipeline(task='sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', device=device)
    if type(generated_sample) == str:
        result = classification(generated_sample)[0]
        score = result['score']
        label = result['label']
        score = 1 - score if label == 'NEGATIVE' else score
        score_shifted = score * shift
        # count occurrence of bonus word
        bonus_count = generated_sample.count(bonus_word)
        score_shifted += bonus_count
        return score_shifted
    elif type(generated_sample) == list:
        results = classification(generated_sample)
        scores = [score_dict['score'] for score_dict in results]
        labels = [score_dict['label'] for score_dict in results]
        scores = [1 - score if label == 'NEGATIVE' else score for score, label in zip(scores, labels)]
        scores = t.tensor(scores, dtype=t.float, device=device)
        scores_shifted = scores * shift
        bonus_appearance = {}
        for samp_idx, sample in enumerate(generated_sample):
            bonus_count = sample.count(bonus_word)
            bonus_appearance[samp_idx] = bonus_count
        scores_shifted += t.tensor([bonus_appearance[samp_idx] for samp_idx in range(len(generated_sample))], dtype=t.float, device=device)
        return scores_shifted


def rfn_neutral_sentiment(generated_sample: Union[str, List[str]]) -> Union[float, Float[Tensor, "batch"]]:
    '''
    Reward function that uses a pre-trained sentiment analysis model to score the sentiment of the generated samples. 
    The function provides reward in proportion to how neutral the sample is, but also provides very high reward if the sentiment is very high.
    This is designed to be an 'exploitable' reward function, in which a token or a set of tokens performs much better on the reward function 
    than the rest of the tokens.
    '''
    classification = pipeline(task='sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', device=device)
    if type(generated_sample) == str:
        result = classification(generated_sample)[0]
        score = result['score']
        label = result['label']
        score = 1 - score if label == 'NEGATIVE' else score
        if score < 0.995:
            score = -t.abs(score - 0.5)
        return score
    elif type(generated_sample) == list:
        results = classification(generated_sample)
        scores = [score_dict['score'] for score_dict in results]
        labels = [score_dict['label'] for score_dict in results]
        scores = [1 - score if label == 'NEGATIVE' else score for score, label in zip(scores, labels)]
        scores = t.tensor(scores, dtype=t.float, device=device)
        scores = t.where(scores >= 0.995, scores, -t.abs(scores - 0.5))
        return scores


def rfn_sentiment_eval(generated_sample: Union[str, List[str]], bonus_word="very") -> Union[float, Float[Tensor, "batch"]]:
    '''
    Reward function that uses a pre-trained sentiment analysis model to score the sentiment of the generated samples. The scores will be
    shifted down by 50%. This is designed to be the companion of the 'exploitable' reward functions rfn_sentiment_capped and rfn_sentiment_uncapped.
    Policies trained with the exploitable reward functions will be tested on this reward function.
    '''
    classification = pipeline(task='sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', device=device)
    if type(generated_sample) == str:
        result = classification(generated_sample)[0]
        score = result['score']
        label = result['label']
        score = 1 - score if label == 'NEGATIVE' else score
        score_shifted = score * 0.5
        return score_shifted
    elif type(generated_sample) == list:
        results = classification(generated_sample)
        scores = [score_dict['score'] for score_dict in results]
        labels = [score_dict['label'] for score_dict in results]
        scores = [1 - score if label == 'NEGATIVE' else score for score, label in zip(scores, labels)]
        scores = t.tensor(scores, dtype=t.float, device=device)
        scores_shifted = scores * 0.5
        return scores_shifted
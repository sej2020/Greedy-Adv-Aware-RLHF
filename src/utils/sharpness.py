import numpy as np
import torch
from torch import Tensor
from pyhessian import hessian
import matplotlib.pyplot as plt
import pickle
from src.utils.replay_memory import ReplayMemory
from src.models.transformers import TransformerWithValueHead
from jaxtyping import Float

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def top_ev(minibatches: ReplayMemory, model: TransformerWithValueHead, obj_fn: callable) -> tuple[list[Float], list[list[Tensor]]]:
    '''
    Computes the top eigenvalues and eigenvectors of the Hessian of the objective function with respect to the model parameters.

    Args:
        minibatches: ReplayMemory object containing the data to be used to compute the Hessian.
        model: TransformerWithValueHead model.
        obj_fn: Objective function to be used to compute the Hessian.

    Returns:
        top_eigenvalues: List of the top eigenvalues of the Hessian.
    '''
    hessian_comp = hessian(model, lambda x: obj_fn(x), dataloader=minibatches, cuda=DEVICE==torch.device('cuda'), minibatch_mod=True)
    top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(maxIter=20, tol=0.01, top_n=3)
    return top_eigenvalues, top_eigenvectors


def obj_landscape(minibatches: ReplayMemory, model: TransformerWithValueHead, obj_fn: callable, top_e_vec: list[Tensor], last_layer_only=False):
    '''
    Perturbs the model parameters along the top eigenvector, evaluates these new models on the objective function
    and plots a cross-section of the parameter landscape with respect to the objective function.

    Args:
        minibatches: ReplayMemory object containing the data to be used to compute the Hessian.
        model: TransformerWithValueHead model.
        obj_fn: Objective function to be used to compute the Hessian.
        top_e_vec: Top eigenvector of the Hessian.
        last_layer_only: Boolean indicating whether to perturb only the last layer of the model.
    
    Returns:
        fig: Figure object containing the plot of the objective function landscape.
        lams: List of the lambda values used to perturb the model parameters.
        obj_list: List of the obj values obtained by evaluating the perturbed models on the objective function.
    '''
    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams = np.linspace(-1.0, 1.0, 41).astype(np.float32)
    obj_list = []

    model.eval()
    model_perb = pickle.loads(pickle.dumps(model))
    model_perb.eval()
    model_perb = model_perb.to(DEVICE)

    # Perturb the model parameters along the top eigenvector and evaluate the objective function
    for lam in lams:
        model_perb = get_params(model, model_perb, top_e_vec, lam, last_layer_only=last_layer_only)
        total_obj = torch.tensor(0.0, device=DEVICE)
        with torch.no_grad():
            for mb in minibatches:
                total_obj += obj_fn(mb, alt_model=model_perb)
        av_obj = total_obj / len(minibatches)
        obj_list.append(av_obj.item())

    del model_perb
    model.train()
    fig = plot_obj_landscape(lams, obj_list)
    return fig, list(lams), obj_list


def plot_obj_landscape(lams: np.ndarray, obj_list: list) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(lams, obj_list)
    ax.set_ylabel("Objective")
    return fig


def get_params(
    model_orig: TransformerWithValueHead,
    model_perb: TransformerWithValueHead, 
    direction: list[Tensor], 
    alpha: Float,
    last_layer_only=False
    ) -> TransformerWithValueHead:
    '''
    Returns a model with the parameters perturbed along the given direction.

    Args:
        model_orig: Original model.
        model_perb: Copy of the original model to be perturbed.
        direction: List of the perturbation directions.
        alpha: Scalar value to scale the perturbation.
        last_layer_only: Boolean indicating whether to perturb only the last layer of the model.
    
    Returns:
        model_perb: Model with the parameters perturbed along the given direction.
    '''
    for (m_orig_name, m_orig), m_perb, d in zip(model_orig.named_parameters(), model_perb.parameters(), direction):
        if 'value' in m_orig_name:
            continue
        if last_layer_only:
            if m_orig_name == "base_model.unembed.W_U" or m_orig_name == "base_model.unembed.b_U":
                m_perb.data = m_orig.data + alpha * d
            else:
                m_perb.data = m_orig.data
        else:
            m_perb.data = m_orig.data + alpha * d

    return model_perb

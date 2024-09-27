import numpy as np
import torch 
from pyhessian import hessian # Hessian computation
from pyhessian.utils import normalization
import copy
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ev_ratio(minibatches, model, loss_fn):
    hessian_comp = hessian(model, loss_fn, dataloader=minibatches, cuda=DEVICE==torch.device('cuda'), minibatch_mod=True)
    top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=50, tol=0.01, top_n=5)
    # print ratio of top eigenvalue to 5th eigenvalue
    return abs(top_eigenvalues[0] / top_eigenvalues[4])


def loss_landscape(minibatches, model, loss_fn, reported="J", label=""):
    assert reported in set(("J_greedy", "J")), "Please choose one of 'J', 'J_greedy' for the loss function to report."
    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    loss_list = []

    # create a copy of the model
    model_perb = copy.copy(model)
    model_perb.eval()
    model_perb = model_perb.to(DEVICE)
    total_loss = torch.tensor(0.0, device=DEVICE)
    for mb in minibatches:
        total_loss += loss_fn(mb, reported=reported, alt_model=model_perb)
    av_loss = total_loss / len(minibatches)
    av_loss.backward()

    v = [p.grad.data for p in model_perb.parameters()]
    v = normalization(v)
    model_perb.zero_grad()

    for lam in lams:
        model_perb = get_params(model, model_perb, v, lam)
        total_loss = torch.tensor(0.0, device=DEVICE)
        with torch.no_grad():
            for mb in minibatches:
                total_loss += loss_fn(mb, reported=reported, alt_model=model_perb)
        av_loss = total_loss / len(minibatches)
        loss_list.append(av_loss.item())

    del model_perb
    fig = plot_loss_landscape(lams, loss_list, label)
    return fig

def plot_loss_landscape(lams, loss_list, label):
    plt.plot(lams, loss_list, label=label)
    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title('Loss landscape perturbed based on gradient direction')
    if label:
        plt.legend()
    fig = plt.gcf()
    return fig


def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

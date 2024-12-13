import numpy as np
import torch 
from pyhessian import hessian
import matplotlib.pyplot as plt
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def top_ev(minibatches, model, loss_fn):
    hessian_comp = hessian(model, lambda x: loss_fn(x), dataloader=minibatches, cuda=DEVICE==torch.device('cuda'), minibatch_mod=True)
    top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(maxIter=20, tol=0.01, top_n=3)
    return top_eigenvalues, top_eigenvectors


def loss_landscape(minibatches, model, loss_fn, top_e_vec, layers=None):
    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams = np.linspace(-1.0, 1.0, 41).astype(np.float32)
    loss_list = []

    model.eval()
    model_perb = pickle.loads(pickle.dumps(model))
    model_perb.eval()
    model_perb = model_perb.to(DEVICE)

    for lam in lams:
        model_perb = get_params(model, model_perb, top_e_vec, lam, layers=layers)
        total_loss = torch.tensor(0.0, device=DEVICE)
        with torch.no_grad():
            for mb in minibatches:
                total_loss += loss_fn(mb, alt_model=model_perb)
        av_loss = total_loss / len(minibatches)
        loss_list.append(av_loss.item())

    del model_perb
    model.train()
    fig = plot_loss_landscape(lams, loss_list)
    return fig, list(lams), loss_list


def plot_loss_landscape(lams, loss_list):
    fig, ax = plt.subplots()
    ax.plot(lams, loss_list)
    ax.set_ylabel("Loss")
    return fig


def get_params(model_orig,  model_perb, direction, alpha, layers=None):
    for (m_orig_name, m_orig), m_perb, d in zip(model_orig.named_parameters(), model_perb.parameters(), direction):
        if 'value' in m_orig_name:
            continue
        if layers:
            if layers == "unembed":
                if m_orig_name == "base_model.unembed.W_U" or m_orig_name == "base_model.unembed.b_U":
                    m_perb.data = m_orig.data + alpha * d
                else:
                    m_perb.data = m_orig.data
            elif layers == "mlp":
                if m_orig_name == "base_model.blocks.11.mlp.W_out" or m_orig_name == "base_model.blocks.11.mlp.b_out":
                    m_perb.data = m_orig.data + alpha * d
                else:
                    m_perb.data = m_orig.data
            else:
                raise ValueError("Please choose one of 'unembed', 'mlp' for the layers argument.")
        else:
            m_perb.data = m_orig.data + alpha * d

    return model_perb

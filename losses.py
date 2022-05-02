import torch
from sde import VPSDE, VESDE, subVPSDE
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise


def get_score_fn(sde, model, train=True, continuous=True):

  if not train:
    model.eval()
  model_fn = model

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
    def score_fn(x, adj, flags, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        score = model_fn(x, adj, flags)
        std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
      else:
        raise NotImplementedError(f"Discrete not supported")
      score = -score / std[:, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, adj, flags, t):
      if continuous:
        score = model_fn(x, adj, flags)
      else:  
        raise NotImplementedError(f"Discrete not supported")
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

  return score_fn


def get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=False, continuous=True, 
                    likelihood_weighting=False, eps=1e-5):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model_x, model_adj, x, adj):

    score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=train, continuous=continuous)

    t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
    flags = node_flags(adj)

    z_x = gen_noise(x, flags, sym=False)
    mean_x, std_x = sde_x.marginal_prob(x, t)
    perturbed_x = mean_x + std_x[:, None, None] * z_x
    perturbed_x = mask_x(perturbed_x, flags)

    z_adj = gen_noise(adj, flags, sym=True) 
    mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
    perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
    perturbed_adj = mask_adjs(perturbed_adj, flags)

    score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)
    score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)

    if not likelihood_weighting:
      losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
      losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)

      losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
      losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

    else:
      g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
      losses_x = torch.square(score_x + z_x / std_x[:, None, None])
      losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

      g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
      losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
      losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj

    return torch.mean(losses_x), torch.mean(losses_adj)

  return loss_fn
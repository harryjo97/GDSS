import torch
from toy_sde import VPSDE, VESDE, subVPSDE


def get_score_fn(sde, model, train=True, continuous=True):

  if not train:
    model.eval()
  model_fn = model

  if isinstance(sde, VPSDE) or isinstance(sde, geoVPSDE):
    def score_fn(x, t):
      if continuous:
        score = model_fn(x)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        raise NotImplementedError(f"Discrete not supported")
      score = -score / std[:, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, t):
      if continuous:
        score = model_fn(x)
      else:  
        raise NotImplementedError(f"Discrete not supported")
      return score
  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

  return score_fn


def get_sde_loss_fn(sde1, sde2, train=True, reduce_mean=False, continuous=True, 
                    likelihood_weighting=False, eps=1e-5):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model1, model2, x):

    score_fn1 = get_score_fn(sde1, model1, train=train, continuous=continuous)
    score_fn2 = get_score_fn(sde2, model2, train=train, continuous=continuous)

    t = torch.rand(x.shape[0], device=x.device) * (sde1.T - eps) + eps

    z1 = torch.randn((x.shape[0], 1), device=x.device)
    mean1, std1 = sde1.marginal_prob(x[:,0].unsqueeze(-1), t)
    perturbed1 = mean1 + std1[:, None] * z1

    z2 = torch.randn((x.shape[0], 1), device=x.device)
    mean2, std2 = sde2.marginal_prob(x[:,1].unsqueeze(-1), t)
    perturbed2 = mean2 + std2[:, None] * z2

    perturbed = torch.cat([perturbed1, perturbed2], dim=-1)

    score1 = score_fn1(perturbed, t)
    score2 = score_fn2(perturbed, t)

    if not likelihood_weighting:
      losses1 = torch.square(score1 * std1[:, None] + z1)
      losses1 = reduce_op(losses1.reshape(losses1.shape[0], -1), dim=-1)

      losses2 = torch.square(score2 * std2[:, None] + z2)
      losses2 = reduce_op(losses2.reshape(losses2.shape[0], -1), dim=-1)

    else:
      raise NotImplementedError(f'Likelihood Weighting not implemented.')

    return torch.mean(losses1), torch.mean(losses2)

  return loss_fn

import torch
import numpy as np
import abc
import pdb
from tqdm import tqdm, trange

from toy_losses import get_score_fn
from toy_sde import VPSDE, VESDE, subVPSDE


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, idx, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.idx = idx

  def update_fn(self, x, t):
    y = x[:, self.idx].unsqueeze(-1)
    dt = -1. / self.rsde.N
    z = torch.randn_like(y)
    drift, diffusion = self.rsde.sde(x, y, t)
    y_mean = y + drift * dt
    y = y_mean + diffusion[:, None] * np.sqrt(-dt) * z
    return y, y_mean


class LangevinCorrector(Corrector):
  def __init__(self, idx, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.idx = idx

  def update_fn(self, x, t):
    sde = self.sde
    target_snr = self.snr
    seps = self.scale_eps

    y = x[:, self.idx].unsqueeze(-1)

    if isinstance(sde, VPSDE) or isinstance(sde, geoVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(self.n_steps):
      grad = self.score_fn(x, t)
      noise = torch.randn_like(y)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      y_mean = y + step_size[:, None] * grad
      y = y_mean + torch.sqrt(step_size * 2)[:, None] * noise * seps

    return y, y_mean

def get_pc_sampler(sde1, sde2, shape, predictor='Euler', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, sampling_steps=1,
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model1, model2):

    score_fn1 = get_score_fn(sde1, model1, train=False, continuous=continuous)
    score_fn2 = get_score_fn(sde2, model2, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor 
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj1 = predictor_fn(0, sde1, score_fn1, probability_flow)
    corrector_obj1 = corrector_fn(0, sde1, score_fn1, snr, scale_eps, n_steps)

    predictor_obj2 = predictor_fn(1, sde2, score_fn2, probability_flow)
    corrector_obj2 = corrector_fn(1, sde2, score_fn2, snr, scale_eps, n_steps)

    with torch.no_grad():
      # Initial sample
      x = torch.randn(shape, device=device)

      diff_steps = sde2.N
      timesteps = torch.linspace(sde2.T, eps, diff_steps, device=device)

      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t

        x1, x1_mean = corrector_obj1.update_fn(x, vec_t)
        x2, x2_mean = corrector_obj2.update_fn(x, vec_t)
        x = torch.cat([x1, x2], dim=-1)

        x1, x1_mean = predictor_obj1.update_fn(x, vec_t)
        x2, x2_mean = predictor_obj2.update_fn(x, vec_t)
        x = torch.cat([x1, x2], dim=-1)
        x_mean = torch.cat([x1_mean, x2_mean], dim=-1)
      print(' ',end='')

      return (x_mean if denoise else x)


  return pc_sampler

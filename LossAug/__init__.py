from torch import nn
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from pytti import *

class Loss(nn.Module):
  def __init__(self, weight, stop, name):
    super().__init__()
    #self.register_buffer('weight', torch.as_tensor(weight))
    #self.register_buffer('stop', torch.as_tensor(stop))
    self.weight = weight
    self.stop = stop
    self.input_axes = ('n','s','y','x')
    self.name = name
    self.enabled = True

  def get_loss(self, input, img):
    raise NotImplementedError
  
  def set_enabled(self, enabled):
    self.enabled = enabled

  def set_weight(self, weight):
    self.weight = weight
  def set_stop(self, stop):
    self.stop = stop

  def __str__(self):
    return self.name

  def forward(self, input, img, device = DEVICE):
    if not self.enabled or self.weight in [0,'0']:
      return 0, 0

    # Python floats, not CUDA tensors: no blocking host->device copy per loss per step
    weight = float(parametric_eval(self.weight))
    stop   = float(parametric_eval(self.stop))
    loss_raw = self.get_loss(input, img)
    loss  =  loss_raw * float((weight > 0) - (weight < 0))
    return abs(weight) * replace_grad(loss, loss.clamp(min=stop)), loss_raw

from pytti.LossAug.TVLoss import TVLoss
from pytti.LossAug.MSELoss import MSELoss
from pytti.LossAug.OpticalFlowLoss import OpticalFlowLoss, TargetFlowLoss
from pytti.LossAug.DepthLoss import DepthLoss
from pytti.LossAug.EdgeLoss import EdgeLoss
from pytti.LossAug.LatentLoss import LatentLoss
from pytti.LossAug.HSVLoss import HSVLoss
LOSS_DICT = {'edge':EdgeLoss, 'depth':DepthLoss}

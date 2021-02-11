import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

class ResLayer(nn.Module):

  def __init__(self, in_dim, out_dim, reduc, conv_reduce):

    super(ResLayer, self).__init__()

    stride = int(reduc)+1

    self.conv1 = nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)

    if in_dim!=out_dim:
      if conv_reduce:
        self.reduc_op = nn.Sequential(
          nn.Conv2d(in_dim, out_dim, 1, stride, 0, bias=False),
          nn.BatchNorm2d(out_dim))
      else:
        pad = abs(out_dim-in_dim)//2
        rem = abs(out_dim-in_dim)%2
        self.reduc_op = nn.Sequential(
          nn.AvgPool2d(1, stride, 0),
          nn.ZeroPad2d((0,0,0,0,pad,pad+rem,0,0)))
    else:
      self.reduc_op = nn.Identity()

  def forward(self, x):

    net = f.relu(self.bn1(self.conv1(x)))
    net = self.bn2(self.conv2(net))
    net = f.relu(net+self.reduc_op(x))

    return net


class ResBlock(nn.Module):

  def __init__(self, in_dim, out_dim, n_layer, reduction, conv_reduce):

    super(ResBlock, self).__init__()

    layers = []

    for i in range(n_layer):

      if i==0 and reduction:
        reduc = True
      else:
        reduc = False
        in_dim = out_dim

      layers.append(ResLayer(in_dim, out_dim, reduc, conv_reduce))

    self.layers = nn.Sequential(*layers)

  def forward(self, net):

    net = self.layers(net)

    return net

class ResNet(nn.Module):

  def __init__(self, dim=16, n_layer=3, conv_reduce=False, n_block=3, multiplier=2, n_cls=10, ch=1, suppress=False):

    super(ResNet, self).__init__()

    self.init_layer = nn.Sequential(
      nn.Conv2d(ch, dim, 3, 1, 1, bias=False),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(dim))

    old_dim = dim

    blocks = []

    for i in range(n_block):

      if i==0:
        mult = 1
        reduc = False
      else:
        mult = multiplier
        reduc = True

      new_dim = old_dim*mult
      blocks.append(ResBlock(old_dim, new_dim, n_layer, reduc, conv_reduce))
      old_dim = new_dim

    self.blocks = nn.Sequential(*blocks)
    self.linear = nn.Linear(new_dim, n_cls)

    params = 0


    for name, param in self.named_parameters():
      if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
        torch.nn.init.kaiming_normal_(param.weight)
      shape = param.shape
      size = 1
      for dim in shape:
        size*=dim
      params+=size

    if not suppress:

      print('param count:', params)

      total_layers = (n_layer*2)*n_block+2
      print('num layers:', total_layers)

  def forward(self, x, with_feats=False):

    net = self.init_layer(x)

    net = self.blocks(net)
    net = torch.mean(net, dim=-1).squeeze()
    net = torch.mean(net, dim=-1).squeeze()

    logits = self.linear(net.reshape(net.shape[0],-1))
    probs = f.softmax(logits, dim=-1)

    if with_feats:
      return logits, probs, net
    else:
      return logits, probs
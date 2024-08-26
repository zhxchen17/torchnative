#!/usr/bin/env python3
import torch

import torch.export._trace

class Module(torch.nn.Module):
    def forward(self, x, y):
        z = (x * y).sum(dim=0)
        w = (x / y).sum(dim=0)
        return z - w

m = Module()
inputs = (torch.randn(3, 4, 5), torch.randn(3, 4, 5))
ep = torch.export._trace._export_for_training(m, inputs)

torch._inductor.aot_compile(ep.module(), inputs)

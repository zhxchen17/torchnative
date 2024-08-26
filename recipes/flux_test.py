#!/usr/bin/env python3
import sys
import unittest.mock

import torch

import flux.util
import flux.cli

assert len(sys.argv) == 3
fmodel_path, ae_decoder_path = sys.argv[1:]

def custom_load_ae_compiled(*args, **kwargs):
    ae = flux.util.load_ae(*args, **kwargs)

    ae_decoder = torch._export.aot_load(ae_decoder_path, 'cuda')
    class DecoderCompiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._ = ae_decoder

        def forward(self, *args, **kwargs):
            return self._(*args, **kwargs)

        def cpu(self):
            self._ = None

    ae.decoder = DecoderCompiled()
    return ae

def custom_load_flow_model_compiled(*args, **kwargs):
    fmodel = torch._export.aot_load(fmodel_path, 'cuda')

    class FluxCompiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._ = fmodel

        def forward(self, *args, **kwargs):
            return self._(*args, **kwargs)

        def cpu(self):
            self._ = None

    return FluxCompiled()


print("testing compiled models...")
with unittest.mock.patch("flux.cli.load_ae", custom_load_ae_compiled), unittest.mock.patch("flux.cli.load_flow_model", custom_load_flow_model_compiled):
    flux.cli.main(height=256, width=256, offload=True)

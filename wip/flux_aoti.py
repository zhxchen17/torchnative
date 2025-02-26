# %pip install torch flux

import unittest.mock
from contextlib import contextmanager

import flux.cli

import flux.util

import torch
import torch.export._trace
import torch.utils._pytree as pytree


def copy_tensors(inputs):
    return pytree.tree_map_only(torch.Tensor, torch.clone, inputs)


def add_sampling_hook(module, samples):
    def _(module, args, kwargs):
        samples.append(copy_tensors((args, kwargs)))

    return module.register_forward_pre_hook(_, prepend=True, with_kwargs=True)


@contextmanager
def sample_inputs(module, samples):
    handle = add_sampling_hook(module, samples)
    try:
        yield
    finally:
        handle.remove()


def dump_samples(samples):
    summary = {}
    for sample in samples:
        flats, spec = pytree.tree_flatten(sample)
        sizes = tuple(x.size() for x in flats)
        if (str(spec), str(sizes)) not in summary:
            summary[(str(spec), str(sizes))] = sample
    return summary


ae = None
ae_decoder_samples = []
fmodel = None
fmodel_samples = []
handles = []


def custom_load_ae(*args, **kwargs):
    global ae
    ae = flux.util.load_ae(*args, **kwargs)
    handles.append(add_sampling_hook(ae.decoder, ae_decoder_samples))
    return ae


def custom_load_flow_model(*args, **kwargs):
    global fmodel
    fmodel = flux.util.load_flow_model(*args, **kwargs)
    handles.append(add_sampling_hook(fmodel, fmodel_samples))
    return fmodel


with unittest.mock.patch("flux.cli.load_ae", custom_load_ae), unittest.mock.patch(
    "flux.cli.load_flow_model", custom_load_flow_model
):
    flux.cli.main(height=256, width=256, offload=True)
    for h in handles:
        h.remove()
    handles.clear()

print("processing ae model...")
with torch.inference_mode(), torch.autocast(
    device_type=torch.device("cuda").type, dtype=torch.bfloat16
):

    vals = copy_tensors(ae_decoder_samples[0])

    def create_dynamic_shape(x):
        col = {}
        for i in range(len(x.shape)):
            col[i] = torch.export.Dim.AUTO
        return col

    dynamic_shap = pytree.tree_map_only(
        torch.Tensor, lambda x: create_dynamic_shape(x), vals
    )[0]

    ae_decoder_ep = torch.export.export(
        ae.decoder,
        *copy_tensors(ae_decoder_samples[0]),
        dynamic_shapes=dynamic_shap,
        strict=False,
    )
    ae_decoder_path = torch._inductor.aot_compile(
        ae_decoder_ep.module(), *ae_decoder_ep.example_inputs
    )

print("processing flux model...")
with torch.inference_mode():
    fmodel.cuda()
    vals_fmodel = copy_tensors(fmodel_samples[0])

    def create_dynamic_shape_v2(x):
        col = {}
        for ix, i in enumerate(x.shape):
            col[ix] = torch.export.Dim.AUTO
        return col

    dynamic_shap_v2 = pytree.tree_map_only(
        torch.Tensor, lambda x: create_dynamic_shape_v2(x), vals_fmodel
    )

    fmodel_ep = torch.export.export(
        fmodel, *vals_fmodel, dynamic_shapes=dynamic_shap_v2[1], strict=False
    )
    fmodel_path = torch._inductor.aot_compile(
        fmodel_ep.module(), *fmodel_ep.example_inputs
    )

print(f"python flux_test.py {fmodel_path} {ae_decoder_path}")

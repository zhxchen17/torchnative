#!/usr/bin/env python3
# %pip install torch openai-whisper

from contextlib import contextmanager
import copy
import json
import subprocess
import os
import pathlib
import shutil

import torch
import torch.utils._pytree as pytree
from torch.export import Dim
import torch.export._trace

import whisper

DEVICE = "cpu"
MODE = "static"

model = whisper.load_model("base")
audio = whisper.load_audio("./tests_jfk.flac")
audio = torch.from_numpy(audio)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
options = whisper.DecodingOptions()

cache_ctx = Dim("cache_ctx", max=446)

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

class MelModule(torch.nn.Module):
    def forward(self, audio):
        return whisper.log_mel_spectrogram(audio)

encoder_samples = []
decoder_samples = []

with sample_inputs(model.encoder, encoder_samples):
    with sample_inputs(model.decoder, decoder_samples):
        result = whisper.decode(model, mel, options)

mel_ep = torch.export._trace._export_for_training(
    MelModule(),
    (audio,),
    strict=False,
)

fqns = {module: name for name, module in model.decoder.named_modules()}
mods = dict(model.decoder.named_modules())

class DecoderWithKVCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, x, xa, kv_cache):
        kv_cache_layers = {mods[k]: v for k, v in kv_cache.items()}
        kv_cache_layers, hooks = model.install_kv_cache_hooks(kv_cache_layers)
        try:
            return self.decoder(x, xa, kv_cache_layers), {fqns[k]: v for k, v in kv_cache_layers.items()}
        finally:
            for _ in hooks:
                _.remove()

encoder_ep = torch.export.export(model.encoder, *encoder_samples[0], strict=False)

detect_ep = torch.export._trace._export_for_training(
    model.decoder,
    *decoder_samples[0],
    strict=False,
)

prefill_ep = torch.export._trace._export_for_training(
    DecoderWithKVCache(),
    *decoder_samples[1],
    strict=False,
)

decoder_args, decoder_kwargs = decoder_samples[2]
decoder_kwargs['kv_cache'] = {fqns[k]: v for k, v in decoder_kwargs['kv_cache'].items()}

def generate_dynamic_shapes(layer: torch.nn.Module):
    if isinstance(layer, whisper.model.MultiHeadAttention):
        kv_cache_dynamic_shapes[layer.key] = {1: cache_ctx}
        kv_cache_dynamic_shapes[layer.value] = {1: cache_ctx}

kv_cache_dynamic_shapes = {}
model.decoder.apply(generate_dynamic_shapes)
for n, m in model.decoder.named_modules():
    if "cross_attn.key" in n:
        kv_cache_dynamic_shapes[m][1] = None
    if "cross_attn.value" in n:
        kv_cache_dynamic_shapes[m][1] = None

dynamic_shapes = {"x": None, "xa": None, "kv_cache": {fqns[k]: v for k, v in kv_cache_dynamic_shapes.items()}}

decoder_ep = torch.export._trace._export_for_training(
    DecoderWithKVCache(),
    decoder_args,
    decoder_kwargs,
    dynamic_shapes=dynamic_shapes,
    strict=False,
)

if MODE == "dynamic":
    with torch.no_grad():
        encoder_so_path = torch._inductor.aot_compile(encoder_ep.module(), *encoder_ep.example_inputs)

    with torch.no_grad():
        prefill_so_path = torch._inductor.aot_compile(prefill_ep.module(), *prefill_ep.example_inputs)

    with torch.no_grad():
        decoder_so_path = torch._inductor.aot_compile(decoder_ep.module(), *decoder_ep.example_inputs)

    manifest = {
        "encoder_so_path": encoder_so_path,
        "prefill_so_path": prefill_so_path,
        "decoder_so_path": decoder_so_path,
    }

elif MODE == "static":
    generated = pathlib.Path(os.path.dirname(__file__)) / "generated"
    if os.path.exists(generated / "decoder.so") and os.path.exists(generated / "prefill.so") and os.path.exists(generated / "encoder.so") and os.path.exists(generated / "detect.so") and os.path.exists(generated / "mel.so"):
        print("model libraries are already generated.")
    else:
        mel_so_path = torch._inductor.aot_compile(
            mel_ep.module(),
            *mel_ep.example_inputs,
            options={"pattern_matcher": False}  # pattern matcher has a bug here.
        )
        mel_so_json = mel_so_path.replace(".so", ".json")

        with torch.no_grad():
            detect_so_path = torch._inductor.aot_compile(
                detect_ep.module(),
                *detect_ep.example_inputs,
            )

            prefill_so_path = torch._inductor.aot_compile(
                prefill_ep.module(),
                *prefill_ep.example_inputs,
            )

            decoder_so_path = torch._inductor.aot_compile(
                decoder_ep.module(),
                *decoder_ep.example_inputs,
            )

            encoder_so_path = torch._inductor.aot_compile(
                encoder_ep.module(),
                *encoder_ep.example_inputs,
            )

            assert DEVICE == "cpu"
            if os.path.exists(generated):
                shutil.rmtree(generated)
            os.mkdir(generated)

            shutil.copyfile(decoder_so_path, generated / "decoder.so")
            shutil.copyfile(prefill_so_path, generated / "prefill.so")
            shutil.copyfile(encoder_so_path, generated / "encoder.so")
            shutil.copyfile(detect_so_path, generated / "detect.so")
            shutil.copyfile(mel_so_path, generated / "mel.so")
            shutil.copyfile(mel_so_json, generated / "mel.json")

    manifest = {
        "prefill_so_path": (generated / "prefill.so").as_posix(),
        "decoder_so_path": (generated / "decoder.so").as_posix(),
        "encoder_so_path": (generated / "encoder.so").as_posix(),
        "detect_so_path": (generated / "detect.so").as_posix(),
        "mel_so_path": (generated / "mel.so").as_posix(),
        "mel_so_json": (generated / "mel.json").as_posix(),
    }

else:
    assert False, f"unknown mode: {MODE}"

filename = f"manifest_{DEVICE}_{MODE}.json"

with open(filename, "w") as f:
    print("Wrting the following to", filename)
    print(manifest)
    json.dump(manifest, f)

# ISSUE 1: dynamic shape warnings.
# ISSUE 2: export on AttrProxy.

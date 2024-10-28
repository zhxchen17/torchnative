#!/usr/bin/env python3
# %pip install torch openai-whisper

from contextlib import contextmanager
import copy
import json
import os
import pathlib
import shutil
import glob
import filecmp

import torch
import torch.utils._pytree as pytree
from torch.export import Dim
import torch.export._trace

import whisper

DEVICE = "cuda"

model = whisper.load_model("base").to(DEVICE)
audio = whisper.load_audio("./tests_jfk.flac")
audio = torch.from_numpy(audio)
audio = whisper.pad_or_trim(audio).to(DEVICE)
mel = whisper.log_mel_spectrogram(audio)
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

mel_ep = torch.export._trace._export_for_training(
    MelModule(),
    (audio,),
    strict=False,
)

encoder_samples = []
decoder_samples = []

with sample_inputs(model.encoder, encoder_samples):
    with sample_inputs(model.decoder, decoder_samples):
        result = whisper.decode(model, mel, options)

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

class PackageBuilder:
    TEMPLATE = """

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow;

use tempfile::{tempdir, TempDir};

pub struct Model<'a> {
    pub device: String,
    pub library: String,
    pub directory: &'a Path,
    pub graph: Option<String>,
}

pub struct ModelPackage {
    devices: HashMap<String, String>,
    libraries: HashMap<String, String>,
    graphs: HashMap<String, String>,
    root: TempDir,
}

impl ModelPackage {
    fn new(root: TempDir) -> Self {
        ModelPackage {
            root,
            devices: HashMap::new(),
            libraries: HashMap::new(),
            graphs: HashMap::new(),
        }
    }

    fn add_library(&mut self, name: String, binary: &'static [u8], device: String, json: Option<&'static [u8]>) {
        self.devices.insert(name.clone(), device);
        let file_path = self.root.path().join(name.clone() + ".so");
        std::fs::write(&file_path, binary).unwrap();
        self.libraries.insert(name.clone(), file_path.to_str().unwrap().to_string());
        if let Some(j) = json {
            let file_path = self.root.path().join(name.clone() + ".json");
            std::fs::write(&file_path, j).unwrap();
            self.graphs.insert(name.clone(), file_path.to_str().unwrap().to_string());
        }
    }

    fn add_kernel(&mut self, name: String, binary: &'static [u8]) {
        let file_path = self.root.path().join(name.clone() + ".cubin");
        std::fs::write(&file_path, binary).unwrap();
    }

    pub fn get(&self, name: &str) -> Model {
        let library = self.libraries[name].clone();
        let graph = self.graphs.get(name).cloned();
        let device = self.devices[name].clone();
        Model {
            device,
            library,
            directory: self.root.as_ref(),
            graph,
        }
    }

    pub fn close(self) -> anyhow::Result<()> {
        self.root.close()?;
        Ok(())
    }

}

pub fn get_models() -> ModelPackage {
    let root = tempdir().unwrap();
    let mut model_package = ModelPackage::new(root);
{{LIBRARIES}}
{{KERNELS}}
    model_package
}
"""
    def __init__(self, device):
        self.device = device
        self.root = pathlib.Path(os.path.dirname(__file__)) / "build" / device
        self.libraries = {}
        self.graphs = {}
        self.kernels = []

    def add_library(self, name, so_path):
        assert os.path.exists(so_path)
        tmp = os.path.dirname(so_path)
        self.libraries[name] = so_path
        json_path = so_path.replace(".so", ".json")
        if os.path.exists(json_path):
            self.graphs[name] = json_path
        if self.device == "cuda":
            self.kernels.extend(glob.glob(tmp + "/*.cubin"))

    def build(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root)
        for name, path in self.libraries.items():
            shutil.copyfile(path, self.root / (name + ".so"))
        for name, path in self.graphs.items():
            shutil.copyfile(path, self.root / (name + ".json"))

        cubins = []
        for path in self.kernels:
            dst = self.root / os.path.basename(path)
            if os.path.exists(dst):
                assert filecmp.cmp(path, dst)
                continue
            cubins.append(os.path.basename(path[:path.rfind(".")]))
            shutil.copyfile(path, dst)

        def graph_arg(device, name):
            if name in self.graphs:
                return f'Some(include_bytes!("../build/{device}/{name}.json"))'
            else:
                return "None"

        device = self.device
        content = PackageBuilder.TEMPLATE.replace(
            "{{LIBRARIES}}",
            "\n".join([f'    model_package.add_library("{name}".to_string(), include_bytes!("../build/{device}/{name}.so"), "{device}".to_string(), {graph_arg(device, name)});' for name in self.libraries])
        )
        content = content.replace(
            "{{KERNELS}}",
            "\n".join([f'    model_package.add_kernel("{name}".to_string(), include_bytes!("../build/{device}/{name}.cubin"));' for name in cubins])
        )
        with open("./src/generated.rs", "w") as f:
            f.write(content)

builder = PackageBuilder(DEVICE)
builder.add_library("decoder", decoder_so_path)
builder.add_library("prefill", prefill_so_path)
builder.add_library("encoder", encoder_so_path)
builder.add_library("detect", detect_so_path)
builder.add_library("mel", mel_so_path)
builder.build()

# ISSUE 1: dynamic shape warnings.
# ISSUE 2: export on AttrProxy.

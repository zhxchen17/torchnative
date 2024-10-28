#!/usr/bin/env python3
import time
import json

import torch
import torch.utils._pytree as pytree
import whisper

DEVICE = "cpu"  # TODO make this an arg.

model = whisper.load_model("base").to(DEVICE)

audio = whisper.load_audio("./nh.m4a")
audio = torch.from_numpy(audio)
audio = whisper.pad_or_trim(audio)

fqns = {module: name for name, module in model.decoder.named_modules()}
mods = dict(model.decoder.named_modules())

def whisper_decode(model, mel):
    assert mel.ndim == 2
    options = whisper.DecodingOptions()
    task = whisper.decoding.DecodingTask(model, options)

    return task.run(mel.unsqueeze(0))[0]

class DecoderCompiled(torch.nn.Module):
    def __init__(self, prefill, decoder, logits):
        super().__init__()
        self.prefill = prefill
        self.decoder = decoder
        self.blocks = model.decoder.blocks
        self.logits = logits
        self.original_decoder = model.decoder

    def forward(self, x, xa, kv_cache=None):
        if kv_cache is None:
            return self.logits(x, xa)
        kv_cache_fqns = {fqns[k]: v for k, v in kv_cache.items()}
        if kv_cache == {}:
            ret, new_kv_cache = self.prefill(x, xa, kv_cache=kv_cache_fqns)
        else:
            ret, new_kv_cache = self.decoder(x, xa, kv_cache=kv_cache_fqns)
        kv_cache.clear()
        kv_cache.update({mods[k]: v for k, v in new_kv_cache.items()})
        return ret

class EncoderCompiled(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
t0 = time.time()
result = whisper_decode(model, mel)
t1 = time.time()
print("eager:", t1 - t0, result.text)

prefill = torch._export.aot_load(f"./build/{DEVICE}/prefill.so", DEVICE)
decoder = torch._export.aot_load(f"./build/{DEVICE}/decoder.so", DEVICE)
encoder = torch._export.aot_load(f"./build/{DEVICE}/encoder.so", DEVICE)
detect = torch._export.aot_load(f"./build/{DEVICE}/detect.so", DEVICE)

model.decoder = DecoderCompiled(prefill, decoder, detect)
model.encoder = EncoderCompiled(encoder)

t0 = time.time()
result = whisper_decode(model, mel)
t1 = time.time()

print("compiled:", t1 - t0)
print(result.text)

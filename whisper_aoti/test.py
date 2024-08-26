#!/usr/bin/env python3
import time
import json

import torch
import torch.utils._pytree as pytree
import whisper

DEVICE = "cpu"  # TODO make this an arg.
MODE = "static"  # TODO make this an arg.

model = whisper.load_model("base")
audio = whisper.load_audio("./tests_jfk.flac")
audio = torch.from_numpy(audio)
audio = whisper.pad_or_trim(audio)
tmp = torch._export.aot_load("./generated/mel.so", DEVICE)
# breakpoint()
# tmp(audio)
if MODE == "static":
    import whisper_aoti
    mel = whisper_aoti.log_mel_spectrogram(audio)
else:
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
# options = whisper.DecodingOptions()

fqns = {module: name for name, module in model.decoder.named_modules()}
mods = dict(model.decoder.named_modules())

def whisper_decode(model, mel, patch=False):
    import types

    assert mel.ndim == 2
    options = whisper.DecodingOptions()
    task = whisper.decoding.DecodingTask(model, options)
    if patch:
        import whisper_aoti
        task = whisper_aoti.make_decoding_task(model.dims)
        # task.tokenizer = whisper_aoti.get_tokenizer()
        # task.logit_filters[0] = whisper_aoti.SuppressBlank(task.tokenizer, task.sample_begin)
        # task.logit_filters[1] = whisper_aoti.SuppressTokens(task._get_suppress_tokens())
        # assert task.decoder.__class__.__name__ == "GreedyDecoder"
        # task.logit_filters[2] = whisper_aoti.ApplyTimestampRules(task.tokenizer, task.sample_begin, task.logit_filters[2].max_initial_timestamp_index)
        # task.decoder = whisper_aoti.GreedyDecoder(options.temperature, task.tokenizer.eot)
        # task.sequence_ranker = whisper_aoti.MaximumLikelihoodRanker(options.length_penalty)
        # task._get_audio_features = types.MethodType(whisper_aoti._get_audio_features, task)
        # task._detect_language = types.MethodType(whisper_aoti._detect_language, task)
        # task._main_loop = types.MethodType(whisper_aoti._main_loop, task)
        # task.run = types.MethodType(whisper_aoti.run, task)

    return task.run(mel.unsqueeze(0))[0]

class DecoderCompiled(torch.nn.Module):
    def __init__(self, prefill, decoder):
        super().__init__()
        self.prefill = prefill
        self.decoder = decoder
        self.blocks = model.decoder.blocks
        self.original_decoder = model.decoder

        if MODE == "static":
            def wrapper(f):
                def _(x, xa, kv_cache):
                    res = f(x, xa, *pytree.tree_leaves(kv_cache))
                    _, out_spec = f.call_spec()
                    return pytree.tree_unflatten(res, pytree.treespec_loads(out_spec))
                return _

            self.prefill = wrapper(self.prefill)
            self.decoder = wrapper(self.decoder)

    def forward(self, x, xa, kv_cache=None):
        if kv_cache is None:
            return self.original_decoder(x, xa)
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
        return self.encoder(x)[0]

with open(f"manifest_{DEVICE}_{MODE}.json", "r") as f:
    manifest = json.load(f)
    if MODE == "dynamic":
        prefill_so_path = manifest["prefill_so_path"]
        decoder_so_path = manifest["decoder_so_path"]
        prefill = torch._export.aot_load(prefill_so_path, DEVICE)
        decoder = torch._export.aot_load(decoder_so_path, DEVICE)
    elif MODE == "static":
        prefill_so_path = manifest["prefill_so_path"]
        decoder_so_path = manifest["decoder_so_path"]
        encoder_so_path = manifest["encoder_so_path"]
        import whisper_aoti
        prefill = whisper_aoti.aot_load(prefill_so_path, DEVICE)
        decoder = whisper_aoti.aot_load(decoder_so_path, DEVICE)
        encoder = whisper_aoti.aot_load(encoder_so_path, DEVICE)
    else:
        assert False, f"Unknown mode: {MODE}"

# t0 = time.time()
# result = whisper_decode(model, mel, options)
# t1 = time.time()
# print("eager:", t1 - t0)

model.decoder = DecoderCompiled(prefill, decoder)
model.encoder = EncoderCompiled(encoder)

t0 = time.time()
result = whisper_decode(model, mel, patch=True)
t1 = time.time()

print("compiled:", t1 - t0)
print(result.text)

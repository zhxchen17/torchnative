# Export & Compile Whisper

![demo](./demo.gif)

## Prerequisites

1. Have a local PyTorch installed.
2. 
``` shell
pip install openai-whisper
```
3.

``` shell
cd $PROJECT/whisper_aoti
```

## Export

```
python ./export.py
```
This command generates 3 shared objects: `encoder.so`, `prefill.so` and `decoder.so`.
For testing purpose, we also generate a json file called `manifest_*.json` which stores
the location for all shared objects.

## Test

``` shell
python ./test.py
```
This script will read the manifest file and replace whisper model with generated shared objects, 
then run compiled version of whisper model.

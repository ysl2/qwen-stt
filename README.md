# qwen-stt

A thin wrapper around MLX-Audio for transcribing audio of arbitrary length with Qwen3-ASR.

## Install

```bash
uv tool install .
```

## Usage

```bash
qwen-stt audio.flac
```

Options:

```bash
qwen-stt audio.flac --format txt --language Chinese
```

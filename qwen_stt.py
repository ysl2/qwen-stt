import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load as load_stt
from mlx_audio.stt.generate import (
    generate_transcription,
    save_as_srt,
    save_as_vtt,
)
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.utils import load_audio
from mlx_audio.vad import load as load_vad
from tqdm import tqdm

MODEL = 'mlx-community/Qwen3-ASR-1.7B-bf16'
VAD_MODEL = 'mlx-community/silero-vad'
SAMPLE_RATE = 16000
TOKENS_PER_SECOND = 12
BASE_TOKENS = 8192


def get_speech_timestamps(audio):
    print('Loading VAD model...')
    vad = load_vad(VAD_MODEL)
    window_size = 512
    state = vad.initial_state(sample_rate=SAMPLE_RATE)
    probabilities = []

    for start in tqdm(range(0, len(audio), window_size), desc='VAD'):
        chunk = audio[start : start + window_size]

        if len(chunk) < window_size:
            chunk = mx.pad(chunk, [(0, window_size - len(chunk))])

        probability, state = vad.feed(
            chunk,
            state,
            sample_rate=SAMPLE_RATE,
        )

        mx.eval(probability, state.state, state.context)
        probabilities.append(float(np.array(probability).item()))

    return vad._probs_to_timestamps(
        np.array(probabilities, dtype=np.float32),
        audio_len=len(audio),
        sample_rate=SAMPLE_RATE,
        threshold=vad.config.threshold,
        min_speech_duration_ms=vad.config.min_speech_duration_ms,
        min_silence_duration_ms=vad.config.min_silence_duration_ms,
        speech_pad_ms=vad.config.speech_pad_ms,
        return_seconds=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--format', default='txt')
    parser.add_argument('--language', default='Chinese')
    args = parser.parse_args()

    print('Loading audio...')
    audio = load_audio(args.audio)
    duration = len(audio) / SAMPLE_RATE
    max_tokens = int(duration * TOKENS_PER_SECOND + BASE_TOKENS)
    output = str(Path(args.audio).with_suffix(''))

    if args.format in ('srt', 'vtt'):
        timestamps = get_speech_timestamps(audio)

        print('Loading STT model...')
        model = load_stt(MODEL)

        segments = []

        for timestamp in tqdm(timestamps, desc='STT'):
            start = timestamp['start']
            end = timestamp['end']

            chunk = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]

            result = model.generate(
                chunk,
                language=args.language,
                max_tokens=int((end - start) * TOKENS_PER_SECOND + BASE_TOKENS),
            )

            segments.append(
                {
                    'start': start,
                    'end': end,
                    'text': result.text,
                }
            )

        result = STTOutput(
            text=' '.join(segment['text'] for segment in segments),
            segments=segments,
            language=args.language,
        )

        {
            'srt': save_as_srt,
            'vtt': save_as_vtt,
        }[args.format](result, output)

    else:
        generate_transcription(
            model=MODEL,
            audio=audio,
            output_path=output,
            format=args.format,
            language=args.language,
            max_tokens=max_tokens,
            verbose=True,
        )

import argparse
from pathlib import Path

import miniaudio
from mlx_audio.stt.generate import generate_transcription

MODEL = 'mlx-community/Qwen3-ASR-1.7B-bf16'
TOKENS_PER_SECOND = 12
BASE_TOKENS = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--format', default='txt')
    parser.add_argument('--language', default='Chinese')
    args = parser.parse_args()

    duration = miniaudio.get_file_info(args.audio).duration
    max_tokens = int(duration * TOKENS_PER_SECOND + BASE_TOKENS)
    output = str(Path(args.audio).with_suffix(''))

    generate_transcription(
        model=MODEL,
        audio=args.audio,
        output_path=output,
        format=args.format,
        language=args.language,
        max_tokens=max_tokens,
        verbose=True,
    )


if __name__ == '__main__':
    main()

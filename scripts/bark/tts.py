import os
import re
import numpy as np
from scipy.io.wavfile import write as write_wav
from bark.generation import (
    preload_models,
    clean_models
)
from bark import generate_audio, SAMPLE_RATE


class TTS:
    def __init__(self, text_prompt, speaker, temperature, silence, voice, low_vram):
        self.text_prompt = text_prompt
        self.speaker = speaker
        self.temperature = temperature
        self.silence = silence
        self.voice = voice
        self.low_vram = low_vram

    def generate(self):
        if self.low_vram:
            preload_models(text_use_gpu=True,
                           text_use_small=True,
                           coarse_use_gpu=True,
                           coarse_use_small=True,
                           fine_use_gpu=True,
                           fine_use_small=True,
                           codec_use_gpu=True,
                           force_reload=False)
        else:
            preload_models(text_use_gpu=True,
                text_use_small=False,
                coarse_use_gpu=True,
                coarse_use_small=False,
                fine_use_gpu=True,
                fine_use_small=False,
                codec_use_gpu=True,
                force_reload=False)
        pieces = []
        # split text_prompt into sentences by punctuation
        sentences = re.split('\[split\]', self.text_prompt)
        silence = np.zeros(int(self.silence * SAMPLE_RATE)).astype(np.float32)
        for sentence in sentences:
            if sentence.strip() != "":
                audio_array = generate_audio(sentence, history_prompt=self.speaker, text_temp=self.temperature)
                pieces += [audio_array, silence.copy()]

        write_wav("bark_generation.wav", SAMPLE_RATE, np.concatenate(pieces))
        clean_models()
        print("Done!")
        return "bark_generation.wav"


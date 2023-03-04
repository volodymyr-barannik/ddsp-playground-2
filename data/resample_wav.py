import os

import librosa
import soundfile as sf

target_sample_rate = 16000
dir_name = "urmp/violin"
for root, dirs, filenames in os.walk(dir_name):
    for filename in filenames:
        abs_path = os.path.abspath(os.path.join(dir_name, filename))

        y, s = librosa.load(abs_path, sr=target_sample_rate) # Downsample 44.1kHz to 8kHz
        sf.write(abs_path, y, target_sample_rate, 'PCM_16')
        #librosa.output.write_wav(abs_path, y, target_sample_rate)
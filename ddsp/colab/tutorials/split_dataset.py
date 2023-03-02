import os

from pydub import AudioSegment
from pydub.utils import make_chunks
import tensorflow as tf
import random

NUMBER_OF_NOISE_AUGMENTATIONS = 0

SAMPLERATE = 48000


def decode_audio(file_name):
    raw_audio = tf.io.read_file(file_name)
    audio, _ = tf.audio.decode_wav(raw_audio)
    return tf.squeeze(audio, axis=-1)


def decode_audio_in_dir(dataset_dir):
    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            abs_path = os.path.abspath(os.path.join(dataset_dir, file))
            try:
                audio_tensor = decode_audio(abs_path)
                print('[+] Success:')
                print(audio_tensor)
            except:
                print('[-] Fail:')
                print(file)


def save(chunks, output_dir, output_file, channel_n, shift):
    out = os.path.join(output_dir, output_file)
    print(f"--\tSaving chunks for {out}")
    for i, audio_chunk in enumerate(chunks):
        clean_chunk_name = out + f"_shift{shift}_{i}.wav"
        audio_chunk.export(clean_chunk_name, format="wav")

        signal = decode_audio(clean_chunk_name)
        for j in range(NUMBER_OF_NOISE_AUGMENTATIONS):
            noise_rms = random.uniform(0, 0.005)
            # noise=np.random.normal(0, noise_rms, audio_chunk!!!.shape[0])
            noise = tf.random.normal(signal.shape, 0, noise_rms)

            noisy_signal = signal + noise
            noisy_signal = tf.expand_dims(noisy_signal, axis=-1)

            noisy_chunk_name = out + f"_channel{channel_n}_shift{shift}_{i}_noise{j}.wav"
            audiostring = tf.audio.encode_wav(noisy_signal, SAMPLERATE)
            tf.io.write_file(noisy_chunk_name, contents=audiostring)


# Splits audio into chunks of constant length. Result is augmented by starting from different timesteps, with the step of shift_step_ms.
def split_file_into_chunks(input_file, output_file, chunk_length_ms=1000, output_dir='./sample_data/chunked_audio/',
                           enable_shift=False, shift_step_ms=50):
    full_audio = AudioSegment.from_file(input_file, format="wav")
    # Set sample rate
    full_audio = full_audio.set_frame_rate(SAMPLERATE)

    if enable_shift is False:
        # Split to mono
        if full_audio.channels > 1:
            splitted_audio = full_audio.split_to_mono()
            for channel in range(len(splitted_audio)):
                chunks = make_chunks(splitted_audio[channel], chunk_length_ms)
                save(chunks=chunks, output_dir=output_dir, output_file=output_file, channel_n=channel,
                     shift=0)
        else:
            chunks = make_chunks(full_audio, chunk_length_ms)
            save(chunks=chunks, output_dir=output_dir, output_file=output_file, channel_n=0, shift=0)
    else:
        for j in range(int(chunk_length_ms / shift_step_ms)):
            shifted_audio = full_audio[j * shift_step_ms:]
            if shifted_audio.channels > 1:
                splitted_audio = shifted_audio.split_to_mono()
                for channel in range(len(splitted_audio)):
                    chunks = make_chunks(splitted_audio[channel], chunk_length_ms)
                    save(chunks=chunks, output_dir=output_dir, output_file=output_file, channel_n=channel,
                         shift=j * shift_step_ms)
            else:
                chunks = make_chunks(shifted_audio, chunk_length_ms)
                save(chunks=chunks, output_dir=output_dir, output_file=output_file, channel_n=0,
                     shift=j * shift_step_ms)


def split_files_into_chunks(input_dir, output_dir, chunk_length_ms):
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filename = os.path.splitext(file)[0]
            abs_path = os.path.abspath(os.path.join(input_dir, file))
            print(f"Splitting {abs_path}.\t\tOutput dir: {output_dir}")
            split_file_into_chunks(abs_path, output_file=filename, chunk_length_ms=chunk_length_ms,
                                   output_dir=output_dir)


split_files_into_chunks(input_dir="E:\\Code\\Projects\\Coursework\\URMP\\Violin",
                        output_dir="E:\\Code\\Projects\\Coursework\\URMP\\ViolinSplit1",
                        chunk_length_ms=4000)

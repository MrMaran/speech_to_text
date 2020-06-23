import pyaudio
from deepspeech import Model
import scipy.io.wavfile as wav
import wave

MODEL_PATH = 'deepspeech-0.7.4-models.pbmm'
SCORER_PATH = 'deepspeech-0.7.4-models.scorer'


def record_voice_stream(stream_file_name):
    stream_format = pyaudio.paInt16  # Sampling size and format
    no_of_channels = 1  # Number of audio channels
    sampling_rate = 16000  # Sampling rate in Hertz
    frames_count = 1024  # Number of frames per buffer
    record_seconds = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=stream_format,
                    channels=no_of_channels,
                    rate=sampling_rate,
                    input=True,
                    frames_per_buffer=frames_count)  # number of frames
    print("Please speak to record your voice")
    frames = [stream.read(frames_count) for i in range(0, int(sampling_rate / frames_count * record_seconds))]
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(stream_file_name, 'wb')
    wf.setnchannels(no_of_channels)
    wf.setsampwidth(p.get_sample_size(stream_format))
    wf.setframerate(sampling_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f'Voice stream file {stream_file_name} is created')


def predict_speech_to_text(stream_file):
    # Initialize the model
    ds = Model(MODEL_PATH)

    # Enable language scorer to improve the accuracy
    ds.enableExternalScorer(SCORER_PATH)
    # You can play with setting the model Beam Width, Scorer language model weight and word insertion weight

    # Use scipy to covert wav file into numpy array
    fs, audio = wav.read(stream_file)
    return ds.stt(audio)


if __name__ == '__main__':
    output_stream_file = 'speech_stream.wav'
    record_voice_stream(output_stream_file)
    print('Start of text prediction')
    print(f'DeepSpeech predicted text: {predict_speech_to_text(output_stream_file)}')

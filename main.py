import pyaudio
import torch
import wave
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__int__()

    def forward(self):
        pass


class Microphone():
    '''
    Represents a simple microphone. To record and play sounds.
    '''
    def __init__(self, width, channels, rate, chunk):
        '''
        Microphone constructor.

        Parameters
        ----------
        width: int
            The width of the data being sampled.
        channels: int
            The number of channels to sample.
        rate: int
            The number of samples to be collected per second.
        chunk: int
            The size of the data to fetch at once.
        '''
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.s = self.p.open(format=self.p.get_format_from_width(width),
                channels=channels,
                rate=rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)

    def __del__(self):
        '''
        Stops the microphone stream and closes pyaudio.
        '''
        self.s.stop_stream()
        self.s.close()
        self.p.terminate()

    def record(self):
        '''
        Records a chunk of sound.
        '''
        return self.s.read(self.chunk)

    def play(self, data):
        '''
        Plays sound on the speaker.

        Parameters
        ----------
        data: bytes
            The data to be played on the speaker.

        TODO: Play sound to virtual microphone.
        '''
        self.s.write(data)

class NoisyIEEE():
    '''
    Contains all of NoisyIEEE data in a varity of datasets.
    '''
    def __init__(self, chunk):
        # Adds all data to dataframes.
        for gen in ['IEEE_Female', 'IEEE_Male']:
            for kind in ['Babble', 'Cafeteria']:
                for level in ['-2dB', '-5dB']:
                    directory = f'NoisyIEEE/{gen}/{kind}/{level}'
                    for _, _, files in os.walk(directory):
                        for filename in files:
                            wav_file = wave.open(f'{directory}/{filename}', mode='rb')
                            for i in range(int(wav_file.getnframes()/chunk)):
                                file_data = wav_file.readframes(chunk)
                                data = [file_data[i:i+2] for i in range(0, 2*CHUNK, 2)]
                            wav_file.close()

def process_data(self, data):
    pass

def predict():
    pass

def train_model():
    pass

if __name__ == '__main__':
    CHUNK = 2
    mic = Microphone(2, 1, 16000, CHUNK)
    data = NoisyIEEE(CHUNK)

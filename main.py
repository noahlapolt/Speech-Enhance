import numpy as np
import pyaudio
import torch
import wave
import json
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

    Assumes width of 2, 2 channels and 16000 sample rate.
    Assumes files are same size.
    '''
    def __init__(self, chunk):
        '''
        Collects all of the IEEE data.
        '''
        # Builds data structure.
        self.all_data = {
            'IEEE_Female':{
                'Babble':{
                    '-2dB':{
                        'Features':[],
                        'Targets':[]
                    },
                    '-5dB':{
                        'Features':[],
                        'Targets':[]
                    }
                }, 
                'Cafeteria':{
                    '-2dB':{
                        'Features':[],
                        'Targets':[]
                    },
                    '-5dB':{
                        'Features':[],
                        'Targets':[]
                    }
                }
            },
            'IEEE_Male':{
                'Babble':{
                    '-2dB':{
                        'Features':[],
                        'Targets':[]
                    },
                    '-5dB':{
                        'Features':[],
                        'Targets':[]
                    }
                }, 
                'Cafeteria':{
                    '-2dB':{
                        'Features':[],
                        'Targets':[]
                    },
                    '-5dB':{
                        'Features':[],
                        'Targets':[]
                    }
                }
            }
        }

        # Adds all data to dataframes.
        for gen in ['IEEE_Female', 'IEEE_Male']:
            for kind in ['Babble', 'Cafeteria']:
                for level in ['-2dB', '-5dB']:
                    directory = f'NoisyIEEE/{gen}/{kind}/{level}'
                    used = []
                    
                    # Goes through all file names in current directory.
                    for _, _, files in os.walk(directory):
                        for f, filename in enumerate(files):
                            prefix = filename.split('_')[0]

                            # Gets clean and noisy data.
                            if prefix not in used:
                                used.append(prefix)
                                clean_file = wave.open(f'{directory}/{prefix}_clean.wav', mode='rb')
                                noisy_file = wave.open(f'{directory}/{prefix}_noisy.wav', mode='rb')
                                for _ in range(int(clean_file.getnframes()/chunk)+1):
                                    noisy_data = process_data(noisy_file.readframes(chunk))
                                    clean_data = process_data(clean_file.readframes(chunk))
                                    
                                    # Adds data to correct area.
                                    self.all_data[gen][kind][level]['Features'].append(noisy_data)
                                    self.all_data[gen][kind][level]['Targets'].append(clean_data)

                                clean_file.close()
                                noisy_file.close()
                                printProgressBar(f/len(files), prefix=f'{directory} Progress:')
                    # Newline.
                    printProgressBar(1, prefix=f'{directory} Progress:')
                    print()

def STFT(data):
    data*np.exp(-1j)


def process_data(data):
    '''
    Preprocesses the data to get more information out of it.

    Parameters
    ----------
    data: list
        A chunk of data as bytes.

    Returns
    -------
    list: The same chunk, but after calculations.
    '''

    # GCCp,q(t,f,k) = np.real((()/())*np.exp())

    # Loops through data.
    # for byte in data:
    #     print(byte)

    return [byte for byte in data]

def predict():
    pass

def train_model():
    pass

def printProgressBar (val, prefix = ''):
    '''
    Loading bar in the console to keep track of tasks.

    Parameters
    ----------
    val: float
        The percent to display.
    prefix: str
        The value before the loading bar.
    '''
    percent = ("{0:." + str(1) + "f}").format(100*val)
    length = 100
    filledLength = int(length*val)
    bar = '█' * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r')

if __name__ == '__main__':
    CHUNK = 1600
    mic = Microphone(2, 1, 16000, CHUNK)
    data = NoisyIEEE(CHUNK)

    print(data.all_data['IEEE_Female']['Babble']['-5dB']['Features'][-2])

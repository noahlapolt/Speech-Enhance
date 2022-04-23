import matplotlib.pyplot as plt
import numpy as np
import argparse
import pyaudio
import signal
import torch
import wave
import os

# Determinds if it can be moved to the gpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Builds mic.
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.s = self.p.open(format=self.p.get_format_from_width(width),
                channels=channels,
                rate=rate,
                input=True,
                output=True,
                frames_per_buffer=chunk,
                stream_callback=self.callback)

        # Turns on mic.
        self.s.start_stream()
        signal.signal(signal.SIGINT, lambda num, frame: self.__del__())
        while self.s.is_active():
            pass

    def __del__(self):
        '''
        Stops the microphone stream and closes pyaudio.
        '''
        self.s.stop_stream()
        self.s.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        '''
        Manages data from microphone.

        Parameters
        ----------
        in_date: bytes
            The raw data from the microphone.
        frame_count: int
            The number of frames in the callback.
        time_info: dict
            Time values related to the callback.
        status: paFlags
            PortAuido callback flag.
        '''
        return (retrive_data(process_data(in_data)), pyaudio.paContinue)


class NoisyIEEE():
    '''
    Contains all of NoisyIEEE data in a varity of datasets.

    Assumes width of 2, 2 channels and 16000 sample rate.
    Assumes files are same size.
    '''
    def __init__(self, chunk, overlap):
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
                                    # Applies overlap.
                                    if noisy_file.tell() != 0:
                                        pos = noisy_file.tell() - overlap
                                        clean_file.setpos(pos)
                                        noisy_file.setpos(pos)

                                    # Reads and processes data.
                                    noisy_data = process_data(noisy_file.readframes(chunk))
                                    clean_data = process_data(clean_file.readframes(chunk))
                                    
                                    # Adds data to correct area.
                                    self.all_data[gen][kind][level]['Features'].append(noisy_data)
                                    self.all_data[gen][kind][level]['Targets'].append(clean_data)

                                clean_file.close()
                                noisy_file.close()
                                printProgressBar(f/len(files), prefix=f'{directory} Progress:')
                    
                    # Shows 100%.
                    printProgressBar(1, prefix=f'{directory} Progress:')
                    print()
                    break
                break
            break


def parse_args():
    '''
    Gets command line arguments for settings.
    '''
    # Create an argument parser that will allow us to capture command line arguments and print help (and default values)
    parser = argparse.ArgumentParser(description='Trains or runs speech enhancment model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', type=str, help='The mode to launch in. Train builds the model from the data. Mic runs it as a speech enhancment Mic.', choices=['train', 'mic'], required=True)
    parser.add_argument('-o', type=int, help='Chunk overlap amount.', default=32)
    parser.add_argument('-c', type=int, help='The size of the chunk to sample.', default=256)
    parser.add_argument('-e', type=int, help='The number of training iterations', default=100)
    parser.add_argument('-l', type=float, help='The learning rate of the model.', default=0.1)
    
    return parser.parse_args()

def process_data(data):
    '''
    Gets the short time fast Fourier transform of the data.

    Parameters
    ----------
    data: list
        A chunk of data as bytes.

    Returns
    -------
    list: The short time fast Fourier transform of the chunk.
    '''

    # Gets chunk of data in integer form.
    chunk = [int.from_bytes(data[i:i+2], 'big') for i in range(0, len(data), 2)]

    # Applies stfft.
    return np.fft.fft(chunk*np.hanning(len(chunk)))

def retrive_data(data):
    '''
    Takes the data from the fft back into byte data.

    Parameters
    ----------
    data: list
        A list of complex numbers.
    '''
    return np.real(np.divide(np.fft.ifft(data), np.hanning(len(data)))).astype(np.ushort).byteswap().tobytes()

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
    length = os.get_terminal_size().columns - len(prefix) - 20
    filledLength = int(length*val)
    bar = '█' * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r')

if __name__ == '__main__':
    args = parse_args()

    if args.m == 'train':
        # Builds the data into a dataframe.
        data = NoisyIEEE(args.c, args.o)
        
        # Displays a portion of the data.
        y = np.real(data.all_data['IEEE_Female']['Babble']['-2dB']['Features'][-2])
        x = range(args.c)

        plt.plot(x, y)
        plt.show()
    else:
        print('Use ctrl+c to stop program.')

        # Creates a microphone to sample data.
        mic = Microphone(2, 1, 16000, args.c)
        

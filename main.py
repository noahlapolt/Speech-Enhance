from utils.ProgBar import ProgBar
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pyaudio
import signal
import torch
import wave
import time
import os

# Determinds if it can be moved to the gpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = np.dtype(np.ushort).newbyteorder('>')
bar = ProgBar()
chunk_size = None
step_size = None
model = None
hanning = None

class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()        
        # Real layers.
        self.r_conv_1 = torch.nn.Conv2d(1, 16, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.r_conv_2 = torch.nn.Conv2d(16, 32, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.r_conv_3 = torch.nn.Conv2d(32, 64, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.r_conv_4 = torch.nn.Conv2d(64, 128, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.r_conv_5 = torch.nn.Conv2d(128, 256, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.r_lstm_1 = torch.nn.LSTM(512, 512, device=device)
        self.r_lstm_2 = torch.nn.LSTM(512, 512, device=device)
        self.r_ff = torch.nn.Linear(512, input_size, device=device)

        # Complex layers.
        self.c_conv_1 = torch.nn.Conv2d(1, 16, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.c_conv_2 = torch.nn.Conv2d(16, 32, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.c_conv_3 = torch.nn.Conv2d(32, 64, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.c_conv_4 = torch.nn.Conv2d(64, 128, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.c_conv_5 = torch.nn.Conv2d(128, 256, (3, 5), stride=(1, 3), padding=(1,3), device=device)
        self.c_lstm_1 = torch.nn.LSTM(512, 512, device=device)
        self.c_lstm_2 = torch.nn.LSTM(512, 512, device=device)
        self.c_ff = torch.nn.Linear(512, input_size, device=device)

        # Data for both.
        self.h0 = torch.randn(1, 11, 512, device=device)
        self.c0 = torch.randn(1, 11, 512, device=device)
        self.relu = torch.nn.ReLU()

        # Loads a save.
        if os.path.exists('model.pt'):
            self.load_state_dict(torch.load('model.pt'))

    def forward(self, data):
        real = data.real
        imag = data.imag

        # Convolution of the data.
        real = self.relu(self.r_conv_1(real))
        real = self.relu(self.r_conv_2(real))
        real = self.relu(self.r_conv_3(real))
        real = self.relu(self.r_conv_4(real))
        real = self.relu(self.r_conv_5(real))

        # LSTM of the data.
        real = torch.reshape(real, (real.size()[0], 11, 512))
        real, (hn, cn) = self.r_lstm_1(real, (self.h0, self.c0))
        real, (hn, cn) = self.r_lstm_2(real[:,-1,:], (hn[:,-1,:], cn[:,-1,:]))
        real = self.relu(self.r_ff(real))
        real = torch.unsqueeze(real, dim=1)

        # Convolution of the data.
        imag = self.relu(self.c_conv_1(imag))
        imag = self.relu(self.c_conv_2(imag))
        imag = self.relu(self.c_conv_3(imag))
        imag = self.relu(self.c_conv_4(imag))
        imag = self.relu(self.c_conv_5(imag))

        # LSTM of the data.
        imag = torch.reshape(imag, (imag.size()[0], 11, 512))
        imag, (hn, cn) = self.c_lstm_1(imag, (self.h0, self.c0))
        imag, (hn, cn) = self.c_lstm_2(imag[:,-1,:], (hn[:,-1,:], cn[:,-1,:]))
        imag = self.relu(self.c_ff(imag))
        imag = torch.unsqueeze(imag, dim=1)

        return torch.cat((real, imag), 1)


class Microphone():
    '''
    Represents a simple microphone. To record and play sounds.
    '''
    def __init__(self, width, channels, rate):
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
        self.p = pyaudio.PyAudio()
        self.frames = int(chunk_size/step_size)
        self.delay = self.frames
        self.buffer = np.zeros((1, self.frames, step_size))

        # Sets the output and input devices. Connects to the virtual cable.
        out_index = 0
        in_index = 0
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if 'VoiceMeeter Input' in device_info['name'] and out_index == 0:
                out_index = device_info['index']
            if 'Microphone (Razer Seiren Mini)' in device_info['name'] and in_index == 0:
                in_index = device_info['index']

        self.s = self.p.open(format=self.p.get_format_from_width(width),
                channels=channels,
                rate=rate,
                input=True,
                output=True,
                frames_per_buffer=step_size,
                stream_callback=self.callback,
                output_device_index=out_index,
                input_device_index=in_index)

        # Turns on mic.
        self.s.start_stream()
        self.running = True
        signal.signal(signal.SIGINT, lambda num, frame: self.toggle_run())
        while self.running:
            pass

        # Cleans up mic.
        self.s.stop_stream()
        self.s.close()
        self.p.terminate()

    def toggle_run(self):
        '''
        Switches mic from recording mode to not recoding mode.
        '''
        self.running = not self.running

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
        if self.delay != 0:
            # Collects inital buffer.
            self.buffer[:,self.frames-self.delay,:] = process_data(in_data)
            self.delay -= 1
            return (in_data, pyaudio.paContinue)
        else:
            # Keeps the rolling buffer.
            self.buffer[:,:self.frames-1,:] = self.buffer[:,1:,:]
            self.buffer[:,-1,:] = process_data(in_data)

            # Predicts and corrects data.
            predict = model.forward(torch.tensor(self.buffer).to(device))
            predict_fft = (predict[0]+predict[1]*1j).detach().numpy()

            return (retrive_data(res), pyaudio.paContinue)


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
        self.all_mix = [[], []]

        # Adds all data to dataframes.
        for gen in ['IEEE_Female', 'IEEE_Male']:
            for kind in ['Babble', 'Cafeteria']:
                for level in ['-2dB', '-5dB']:
                    directory = f'NoisyIEEE/{gen}/{kind}/{level}'
                    used = []
                    bar.prefix = f'{directory} Progress:'
                    bar.val = 0

                    # Goes through all file names in current directory.
                    for _, _, files in os.walk(directory):
                        for f, filename in enumerate(files):
                            prefix = filename.split('_')[0]

                            # Gets clean and noisy data.
                            if prefix not in used:
                                used.append(prefix)
                                clean_file = wave.open(f'{directory}/{prefix}_clean.wav', mode='rb')
                                noisy_file = wave.open(f'{directory}/{prefix}_noisy.wav', mode='rb')
                                for pos in range(0, clean_file.getnframes(), step_size):
                                    # Reads and processes data.
                                    noisy_data = process_data(noisy_file.readframes(chunk))
                                    clean_data = process_data(clean_file.readframes(chunk))[-1,:]

                                    # Moves the data pointer back.
                                    noisy_file.setpos(pos)
                                    clean_file.setpos(pos)

                                    # Stores the data.
                                    self.all_mix[0].append(noisy_data)
                                    self.all_mix[1].append([clean_data.real, clean_data.imag])

                                clean_file.close()
                                noisy_file.close()
                                bar.val = f/len(files)
                    
                    # Shows 100%.
                    bar.val = 1
        self.all_mix = [np.array(self.all_mix[0]), np.array(self.all_mix[1])]


def process_data(data):
    '''
    Gets the short time fast Fourier transform of the data.

    Parameters
    ----------
    data: list
        A chunk of data representing in bytes (1s).

    Returns
    -------
    list: The short time fast Fourier transform of the chunk.
    '''
    # Integer data from bytes.
    res = np.frombuffer(data, dtype=dt)
    chunk = np.zeros(chunk_size)
    chunk[:len(res)] = res

    # Performs short time Fourier transform for each chunk of data.
    res = []
    for i in range(0, chunk_size, step_size):
        res.append(np.fft.fft(chunk[i:i+step_size]*hanning))

    # Resulting data.
    return np.array(res, dtype=np.csingle)

def retrive_data(data):
    '''
    Takes the data from the fft back into byte data.

    Parameters
    ----------
    data: list
        A list of complex numbers.
    '''
    # Gets the real value of the inverted Fourier transform. 
    # Clips first and last value to avoid divide by zero error.
    invert = np.around(np.real(np.fft.ifft(data))[1:-1]/hanning[1:-1])

    # Transforms the inveted piece into bytes. Accounts for missing two vals.
    return np.concatenate(([0], invert, [0])).astype(dt)#.tobytes()

def train_model(data, epochs, lr, data_opt='all'):
    buffer_size = 1000
    feats = torch.split(torch.unsqueeze(torch.tensor(data[0]).to(device), dim=1), 1000, dim=0)
    targs = torch.split(torch.tensor(data[1]).to(device), 1000, dim=0)

    # Model training params.
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    avg_losses = []

    # Randomizes data.
    rands = np.arange(len(feats))
    np.random.seed(100)
    np.random.shuffle(rands)
    np.random.seed()

    # Splits data into train and test.
    train = rands[int(len(rands)-len(rands)/2):]
    test = rands[:-int(len(rands)/2)]

    # Trains for specific number of runs.
    signal.signal(signal.SIGINT, lambda num, frame: evaluate(avg_losses, test, feats, targs))
    last_loss = 0
    for e in range(epochs):
        # Randomizes the data each epoch
        np.random.shuffle(train)

        # Time and losses.
        start_time = time.time()
        losses = []
        for i in train:
            # Makes a prediction.
            outputs = model.forward(feats[i])
            optimizer.zero_grad()

            # Calculates loss and takes action based on it.
            loss = criterion(outputs, targs[i])
            loss.backward()
            optimizer.step()

            # Shows progress.
            losses.append(loss.item())
        
        # Loading bar and saves.
        avg_loss = np.mean(losses)
        bar.prefix = f'Loss: {np.around(avg_loss)} Est Time Left: {np.around((time.time()-start_time)*(epochs-e), decimals=2)}s Change in loss: {np.around(avg_loss-last_loss)}'
        bar.val = e/epochs
        torch.save(model.state_dict(), 'model.pt')
        last_loss = avg_loss
        avg_losses.append(avg_loss)
    
    # Shows 100% and evaluates.
    bar.msg = f'Yo, <@241751545912229888> your AI has finished training! It\'s final loss was {avg_losses[-1]}.'
    bar.prefix = f'Loss: {avg_losses[-1]} Est Time Left: 0'
    bar.val = 1
    evaluate(avg_losses, test, feats, targs)

def evaluate(losses, test, feats, targs):
    print()
    # Saves model.
    torch.save(model.state_dict(), 'model.pt')

    # Gets accuracy on test.
    total = 0
    acuracy = 0
    for i in test:
        pred = model.forward(feats[i])
        pred = pred[:,0,:] + 1j*pred[:,1,:]
        targ = targs[i][:,0,:] + 1j*targs[i][:,1,:]
        
        for k in range(targ.size(0)):
            p = retrive_data(pred[k,:].cpu().detach().numpy())
            t = retrive_data(targ[k,:].cpu().detach().numpy())
            acuracy += np.sum(p == t)

        # Gets sum.
        total += pred.size(0) + pred.size(1)

    print(f'Accuracy: {acuracy/total}')

    # Displays loss over time.
    plt.plot(range(len(losses)), losses)
    plt.show()
    exit(1)

def parse_args():
    '''
    Gets command line arguments for settings.
    '''
    # Create an argument parser that will allow us to capture command line arguments and print help (and default values)
    parser = argparse.ArgumentParser(description='Trains or runs speech enhancment model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', type=str, help='The mode to launch in. Train builds the model from the data. Mic runs it as a speech enhancment Mic.', choices=['train', 'mic'], required=True)
    parser.add_argument('-f', type=int, help='The number of frequency bins.', default=161)
    parser.add_argument('-t', type=int, help='The time frame to include.', default=11)
    parser.add_argument('-e', type=int, help='The number of training iterations', default=100)
    parser.add_argument('-l', type=float, help='The learning rate of the model.', default=0.01)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    chunk_size = args.f*args.t
    step_size = args.f
    model = Model(args.f)
    hanning = np.hanning(step_size)

    if args.m == 'train':
        # Builds the data into a dataframe.
        data = NoisyIEEE(chunk_size)
        train_model(data.all_mix, args.e, args.l)
    else:
        # Creates a microphone to sample data.
        print('Use ctrl+c to stop program.')
        mic = Microphone(2, 1, 16000, chunk_size)
        
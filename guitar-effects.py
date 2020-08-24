import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import Colorblind
import pydub
from pydub.playback import play
import IPython.display as ipd
import os
output_notebook()

# This class defines the core Guitar Effects object. 
# It contains functions to read and write audio files.
# It also contains all the different functions implementing various guitar effects

class GEcore():
    
    def __init__(self):
        self.effectname = ''
        self.audiofilename = ''
        self.framerate = []
        self.signal = []
        self.read_audiofile()
        
    def read_audiofile(self):
        name = input('Enter the audio filename you want to read including the extension: ')
        filename, file_ext = os.path.splitext(name)
        filename = os.getcwd() + '/audiofiles/' + name
        self.audiofilename = filename
        audiofile = pydub.AudioSegment.from_file(filename, file_ext)
        audiofile = audiofile.fade_out(2000)
        self.framerate = audiofile.frame_rate
        songdata = []  # Empty list for holding audio data
        channels = []  # Empty list to hold data from separate channels
        songdata = np.frombuffer(audiofile._data, np.int16)
        for chn in range(audiofile.channels):
            channels.append(songdata[chn::audiofile.channels])  # separate signal from channels
        self.signal = np.sum(channels, axis=0) / len(channels)  # Averaging signal over all channels
        self.signal = self.norm_signal(self.signal)  # normalize signal amplitude
        p_audiofile = self.plot_signal([self.signal])
        show(p_audiofile)
        
    def norm_signal(self, input_signal):
        output_signal = input_signal / np.max(np.absolute(input_signal))
        return output_signal
        
    def plot_signal(self, audio_signal):
        p = figure(plot_width=900, plot_height=500, title='Audio Signal', 
                   x_axis_label='Time (s)', y_axis_label='Amplitude (arb. units)')
        time = np.linspace(0, np.shape(audio_signal)[1] / self.framerate, np.shape(audio_signal)[1])
        m = int(np.shape(audio_signal)[1] / 2000)
        for n in range(np.shape(audio_signal)[0]):
            labels = 'signal ' + str(n + 1)
            p.line(time[0::m], audio_signal[n][0::m], line_color=Colorblind[8][n], 
                   alpha=0.6, legend_label=labels)
        return(p)
    
    def delay(self, input_signal):
        delaytime = int(input('Enter the delay you want to add (> 50ms and < 5000ms): '))
        gain = float(input('Enter the gain (number betweeen 0 and 1): '))
        num = int(delaytime * 1e-3 * self.framerate)
        delaysig = np.roll(input_signal, num)
        delaysig[:num] = 0
        output_signal = input_signal + gain * delaysig
        output_signal = self.norm_signal(output_signal)
        p_delay = self.plot_signal([input_signal, output_signal])
        show(p_delay)
        return output_signal
    
    def flanger(self, input_signal):
        maxdelay = int(input('Enter the maximum delay you want to add (< 15ms): '))
        fflanger = float(input('Enter the frequency of delay oscillation (~ 1Hz): '))
        gain = float(input('Enter the gain (number betweeen 0 and 1): '))
        num = int(maxdelay * 1e-3 * self.framerate)
        output_signal = np.zeros(len(input_signal))
        delaysig = np.zeros(num)
        for n in range(len(input_signal)):
            d = int(0.5 * num * (1 + np.sin(2 * np.pi * fflanger * n / self.framerate)))
            if d < n:
                output_signal[n] = input_signal[n] + gain * input_signal[n-d]
            else:
                output_signal[n] = input_signal[n] 
        output_signal = self.norm_signal(output_signal)
        p_flanger = self.plot_signal([input_signal, output_signal])
        show(p_flanger)
        return output_signal
    
    def overdrive(self, input_signal):
        th = float(input('Enter the threshold (number between 0 and 1): '))
        output_signal = np.zeros(len(input_signal))
        for n in range(len(input_signal)):
            if np.absolute(input_signal[n]) < th:
                output_signal[n] = 2 * input_signal[n]
            if np.absolute(input_signal[n]) >= th:
                if input_signal[n] > 0:
                    output_signal[n] = (3 - (2 - 3 * input_signal[n])**2) / 3
                if input_signal[n] < 0:
                    output_signal[n] = -(3 - (2 - 3 * np.absolute(input_signal[n]))**2) / 3
            if np.absolute(input_signal[n]) > 2 * th:
                if input_signal[n] > 0:
                    output_signal[n] = 1
                if input_signal[n] < 0:
                    output_signal[n] = -1
        output_signal = self.norm_signal(output_signal)
        p_overdrive = self.plot_signal([input_signal, output_signal])
        show(p_overdrive)
        return output_signal
    
    def tremolo(self, input_signal):
        alph = float((input('Enter the amplitude for the modulation (number between 0 and 1): ')))
        modfreq = float(input('Enter modulation frequency (< 20Hz): '))
        output_signal = np.zeros(len(input_signal))
        for n in range(len(input_signal)):
            trem = 1 + alph * np.sin(2 * np.pi * modfreq * n / self.framerate)
            output_signal[n] = trem * input_signal[n]
        output_signal = self.norm_signal(output_signal)
        p_tremolo = self.plot_signal([input_signal, output_signal])
        show(p_tremolo)
        return output_signal
    
    def wahwah(self, input_signal):
        damp = float(input('Enter the damping factor (number between 0 and 1): '))
        minf = float(input('Enter minimum center cutoff frequency (~ 500Hz): '))
        maxf = float(input('Enter the maximum center cutoff frequency (~ 5000Hz): '))
        wahf = float(input('Enter the "wah" frequency (~ 2000Hz): '))
        output_signal = np.zeros(len(input_signal))
        outh = np.zeros(len(input_signal))
        outl = np.zeros(len(input_signal))
        delta = wahf / self.framerate
        centerf = np.concatenate((np.arange(minf, maxf, delta), np.arange(maxf, minf, -delta)))
        while len(centerf) < len(input_signal):
            centerf = np.concatenate((centerf, centerf))
        centerf = centerf[:len(input_signal)]
        f1 = 2 * np.sin(np.pi * centerf[0] / self.framerate)
        outh[0] = input_signal[0]
        output_signal[0] = f1 * outh[0]
        outl[0] = f1 * output_signal[0]
        for n in range(1, len(input_signal)):
            outh[n] = input_signal[n] - outl[n-1] -  2 * damp * output_signal[n-1]
            output_signal[n] = f1 * outh[n] + output_signal[n-1]
            outl[n] = f1 * output_signal[n] + outl[n-1]
            f1 = 2 * np.sin(np.pi * centerf[n] / self.framerate)
        output_signal = self.norm_signal(output_signal)
        p_wahwah = self.plot_signal([input_signal, output_signal])
        show(p_wahwah)
        return output_signal
    
    def octaveup(self, input_signal):
        gain = float(input('Enter gain of octave (number between 0 and 1): '))
        output_signal = input_signal + gain * np.absolute(input_signal)
        output_signal = self.norm_signal(output_signal)
        p_octaveup = self.plot_signal([input_signal, output_signal])
        show(p_octaveup)
        return output_signal

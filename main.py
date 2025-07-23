import tkinter as tk # Para interfaz
from tkinter import ttk, filedialog
import numpy as np
from playsound3 import playsound # Para manejo de sonidos simples
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.fft import fft, ifft # Para análisis de Fourier

T = 0.25 # Duración de pulso
Fs = 32768 # Samplig rate
t = np.linspace(0, T, int(Fs * T), endpoint=False)  # Vector tiempo
fr = np.array([697, 770, 852, 941])  # Frecuencias de filas
fc = np.array([1209, 1336, 1477])  # Frecuencias de columnas
keypad = {
    '1': (0, 0), '2': (0, 1), '3': (0, 2),
    '4': (1, 0), '5': (1, 1), '6': (1, 2),
    '7': (2, 0), '8': (2, 1), '9': (2, 2),
    '*': (3, 0), '0': (3, 1), '#': (3, 2)
}
inverseKeypad = {tuple(v): k for k, v in keypad.items()}

# Pulsar dígitos en un teclado telefónico estándar
def pressDigit(digit):
    if digit in keypad:
        signal = getDigitSound(digit)
        getSinSignalGraph(signal, digit)
        getFrequenciesGraph(signal, digit)
        playSignal(signal)

# Sonido asociado al dígito
def getDigitSound(digit):
    row, col = keypad[digit]
    y1 = np.sin(2 * np.pi * fr[row] * t)
    y2 = np.sin(2 * np.pi * fc[col] * t)
    signal = (y1 + y2) / 2  # DTMF
    return signal * 32767  # Scale to 16-bit audio range

# Obtener gráfica de la señal senoidal asociada (gráfica continua)
def getSinSignalGraph(signal, digit):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(f'Sin Signal for Digit {digit}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Obtener gráfica de las frecuencias que generan la gráfica continua (gráfica discreta)
def getFrequenciesGraph(signal, digit):
    N = len(signal)
    freq = np.fft.fftfreq(N, 1/Fs)
    freq_signal = np.abs(fft(signal))
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:N//2], freq_signal[:N//2])
    plt.title(f'Frequency Spectrum for Digit {digit}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

# Reproducir la señal cargada
def playSignal(signal):
    from scipy.io.wavfile import write
    write('temp.wav', Fs, signal.astype(np.int16))
    playsound('temp.wav')

# Decodificar la señal usando la transformada de Fourier y mostrar los dígitos marcados
def decodeSignal():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        fs, data = wavfile.read(file_path)
        if fs != Fs:
            data = np.interp(np.linspace(0, len(data)/fs, int(Fs*T*len(data)/fs)), 
                            np.arange(len(data))/fs, data)
        N = len(data)
        freq = np.fft.fftfreq(N, 1/Fs)
        freq_signal = np.abs(fft(data))
        digits = []
        segment_size = int(Fs * T)
        for i in range(0, len(data), segment_size):
            segment = data[i:i+segment_size]
            if len(segment) < segment_size:
                break
            freq_seg = np.abs(fft(segment))
            row_freq = fr[np.argmax(freq_seg[np.abs(freq) < 1000]) // (N//Fs)]
            col_freq = fc[np.argmax(freq_seg[np.abs(freq) > 1000]) // (N//Fs)]
            row_idx = np.where(fr == row_freq)[0][0]
            col_idx = np.where(fc == col_freq)[0][0]
            digit = inverseKeypad[(row_idx, col_idx)]
            digits.append(digit)
        return ''.join(digits[:11])  # Return first 11 digits

# Graficar la señal cargada
def graphSignal():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        fs, data = wavfile.read(file_path)
        if fs != Fs:
            data = np.interp(np.linspace(0, len(data)/fs, int(Fs*T*len(data)/fs)), 
                            np.arange(len(data))/fs, data)
        t_loaded = np.linspace(0, len(data)/Fs, len(data))
        plt.figure(figsize=(10, 4))
        plt.plot(t_loaded, data)
        plt.title('Loaded Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()
        digits = decodeSignal()
        print(f"Decoded Digits: {digits}")

# GUI Setup
root = tk.Tk()
root.title("DTMF Signal Generator and Decoder")

# Botones Keypad
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

for i, row in enumerate(['123', '456', '789', '*0#']):
    for j, digit in enumerate(row):
        btn = ttk.Button(frame, text=digit, command=lambda d=digit: pressDigit(d))
        btn.grid(row=i, column=j, padx=5, pady=5)

# Botón de cargar
load_btn = ttk.Button(root, text="Load and Decode Signal", command=graphSignal)
load_btn.grid(row=1, column=0, pady=10)

root.mainloop()
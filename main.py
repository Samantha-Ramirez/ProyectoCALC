import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from playsound3 import playsound
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.fft import fft, fftfreq

# Constantes
T = 0.25  # Duración de cada tono en segundos
Fs = 32768  # Frecuencia de muestreo
t = np.linspace(0, T, int(Fs * T), endpoint=False)  # Vector de tiempo
fr = np.array([697, 770, 852, 941])  # Frecuencias de filas
fc = np.array([1209, 1336, 1477])  # Frecuencias de columnas
keypad = {
    '1': (0, 0), '2': (0, 1), '3': (0, 2),
    '4': (1, 0), '5': (1, 1), '6': (1, 2),
    '7': (2, 0), '8': (2, 1), '9': (2, 2),
    '*': (3, 0), '0': (3, 1), '#': (3, 2)
}
inverseKeypad = {v: k for k, v in keypad.items()}

# Helper function para encontrar la frecuencia más cercana
def findClosestFreq(peak_freq, frequencies):
    return frequencies[np.argmin(np.abs(frequencies - peak_freq))]

# CODIFICACIÓN -------------------------------------
# 1. Pulsar dígitos en un teclado telefónico estándar
def getDigitSound(digit):
    row, col = keypad[digit]
    y1 = np.sin(2 * np.pi * fr[row] * t)
    y2 = np.sin(2 * np.pi * fc[col] * t)
    signal = (y1 + y2) / 2
    return signal * 32767

# Reproducir la señal
def playSignal(signal):
    from scipy.io.wavfile import write
    write('temp.wav', Fs, signal.astype(np.int16))
    playsound('temp.wav')

# Graficar señal senoidal (gráfica continua)
def getSinSignalGraph(signal, digit):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title(f'Señal senoidal de {digit}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()

# Graficar espectro de frecuencias (gráfica discreta)
def getFreqGraph(signal, digit):
    N = len(signal)
    freq = fftfreq(N, 1/Fs)
    freq_signal = np.abs(fft(signal)) / np.sqrt(N)  # Normalización según propiedad DFT
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:N//2], freq_signal[:N//2])
    row, col = keypad[digit]
    plt.axvline(x=fr[row], color='r', linestyle='--', label=f'Frecuencia fila: {fr[row]} Hz')
    plt.axvline(x=fc[col], color='g', linestyle='--', label=f'Frecuencia columna: {fc[col]} Hz')
    plt.title(f'Espectro de frecuencias de {digit}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid()
    plt.show()

# Manejar pulsación de dígitos
def pressDigit(digit):
    if digit in keypad:
        signal = getDigitSound(digit)
        playSignal(signal)
        getSinSignalGraph(signal, digit)
        getFreqGraph(signal, digit)

# DECODIFICACIÓN -------------------------------------
# 2. Decodificar la señal usando la transformada de Fourier y mostrar los dígitos marcados
def decodeLoadedSignal(file_path=None):
    if not file_path:
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        fs, data = wavfile.read(file_path)
        if fs != Fs:
            num_samples = int(len(data) * Fs / fs)
            data = np.interp(np.linspace(0, len(data), num_samples), np.arange(len(data)), data)
            fs = Fs
        segment_size = int(fs * T)
        digits = []
        for i in range(0, len(data), segment_size):
            segment = data[i:i+segment_size]
            if len(segment) < segment_size:
                break
            N = len(segment)
            freq = fftfreq(N, 1/Fs)
            freq_signal = np.abs(fft(segment)) / np.sqrt(N)  # Normalización para magnitud precisa
            row_mask = (freq >= 600) & (freq <= 1000)
            col_mask = (freq >= 1100) & (freq <= 1500)
            if np.any(row_mask) and np.any(col_mask):
                row_peak_freq = freq[row_mask][np.argmax(freq_signal[row_mask])]
                col_peak_freq = freq[col_mask][np.argmax(freq_signal[col_mask])] 
                row_freq = findClosestFreq(row_peak_freq, fr)
                col_freq = findClosestFreq(col_peak_freq, fc)
                row_idx = np.where(fr == row_freq)[0][0]
                col_idx = np.where(fc == col_freq)[0][0]
                digit = inverseKeypad.get((row_idx, col_idx), '?')
                digits.append(digit)
        return ''.join(digits[:11])  # Limitar a 11 dígitos

# Cargar, reproducir y graficar señal
def graphLoadedSignal():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        fs, data = wavfile.read(file_path)
        if fs != Fs:
            num_samples = int(len(data) * Fs / fs)
            data = np.interp(np.linspace(0, len(data), num_samples), np.arange(len(data)), data)
            fs = Fs
        t_loaded = np.linspace(0, len(data)/fs, len(data))
        
        # Reproducir señal cargada
        playSignal(data)
        
        # Graficar señal cargada
        plt.figure(figsize=(10, 4))
        plt.plot(t_loaded, data)
        plt.title('Señal cargada')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.show()
        
        # Añadir espectrograma
        plt.figure(figsize=(10, 4))
        plt.specgram(data, Fs=fs, NFFT=1024, noverlap=512)
        plt.title('Espectrograma de la señal cargada')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.colorbar(label='Intensidad')
        plt.show()
        
        # Decodificar y mostrar dígitos
        digits = decodeLoadedSignal(file_path)
        decoded_label.config(text=f"Dígitos decodificados: {digits}")

# Configuración de la GUI
root = tk.Tk()
root.title("Teléfono")

# Frame para el teclado
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Botones del teclado
for i, row in enumerate(['123', '456', '789', '*0#']):
    for j, digit in enumerate(row):
        btn = ttk.Button(frame, text=digit, command=lambda d=digit: pressDigit(d))
        btn.grid(row=i, column=j, padx=5, pady=5)

# Botón para cargar señal
load_btn = ttk.Button(root, text="Cargar señal de archivo", command=graphLoadedSignal)
load_btn.grid(row=1, column=0, pady=10)

# Etiqueta para mostrar dígitos decodificados
decoded_label = ttk.Label(root, text="Dígitos decodificados: ")
decoded_label.grid(row=2, column=0, pady=10)

root.mainloop()
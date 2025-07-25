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

# CODIFICACIÓN -------------------------------------
# 1. Pulsar dígitos en un teclado telefónico estándar
def pressDigit(digit):
    if digit in keypad:
        signal = getDigitSound(digit)
        playSignal(signal)
        getSinSignalGraph(signal, digit)
        getFreqGraph(signal, digit)

# Obtener sonido asociado al dígito marcado
def getDigitSound(digit):
    row, col = keypad[digit]
    y1 = np.sin(2 * np.pi * fr[row] * t)
    y2 = np.sin(2 * np.pi * fc[col] * t)
    signal = (y1 + y2) / 2
    return signal * 32767

# Reproducir señal
def playSignal(signal):
    from scipy.io.wavfile import write
    write('output.wav', Fs, signal.astype(np.int16))
    playsound('output.wav')

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
    freqSignal = np.abs(fft(signal)) / np.sqrt(N)  # Normalización según propiedad DFT
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:N//2], freqSignal[:N//2])
    row, col = keypad[digit]
    plt.axvline(x=fr[row], color='r', linestyle='--', label=f'Frecuencia fila: {fr[row]} Hz')
    plt.axvline(x=fc[col], color='g', linestyle='--', label=f'Frecuencia columna: {fc[col]} Hz')
    plt.title(f'Espectro de frecuencias de {digit}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid()
    plt.show()

# Obtener recuencia más cercana
def findClosestFreq(peakFreq, frequencies):
    return frequencies[np.argmin(np.abs(frequencies - peakFreq))]

# DECODIFICACIÓN -------------------------------------
# 2. Decodificar la señal usando la transformada de Fourier y mostrar los dígitos marcados
def loadAudioFile(filePath=None):
    if not filePath:
        filePath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if filePath:
        fs, data = wavfile.read(filePath)
        if fs != Fs:
            numSamples = int(len(data) * Fs / fs)
            data = np.interp(np.linspace(0, len(data), numSamples), np.arange(len(data)), data)
            fs = Fs
        return fs, data
    return None, None

def decodeSegment(segment, fs):
    N = len(segment)
    freq = fftfreq(N, 1/Fs)
    freqSignal = np.abs(fft(segment)) / np.sqrt(N)
    rowMask = (freq >= 600) & (freq <= 1000)
    colMask = (freq >= 1100) & (freq <= 1500)
    if np.any(rowMask) and np.any(colMask):
        rowPeakFreq = freq[rowMask][np.argmax(freqSignal[rowMask])]
        colPeakFreq = freq[colMask][np.argmax(freqSignal[colMask])]
        rowFreq = findClosestFreq(rowPeakFreq, fr)
        colFreq = findClosestFreq(colPeakFreq, fc)
        rowIdx = np.where(fr == rowFreq)[0][0]
        colIdx = np.where(fc == colFreq)[0][0]
        return inverseKeypad.get((rowIdx, colIdx), '?')
    return None

def decodeSignal(fs, data):
    segmentSize = int(fs * T)
    digits = []
    for i in range(0, len(data), segmentSize):
        segment = data[i:i+segmentSize]
        if len(segment) < segmentSize:
            break
        digit = decodeSegment(segment, fs)
        if digit:
            digits.append(digit)
    return ''.join(digits[:11])  # Limitar a 11 dígitos

def plotTimeDomain(data, fs):
    tLoaded = np.linspace(0, len(data)/fs, len(data))
    plt.figure(figsize=(10, 4))
    plt.plot(tLoaded, data)
    plt.title('Señal cargada')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()

def plotSpectrogram(data, fs):
    plt.figure(figsize=(10, 4))
    plt.specgram(data, Fs=fs, NFFT=1024, noverlap=512)
    plt.title('Espectrograma de la señal cargada')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(label='Intensidad')
    plt.show()

def updateDecodedLabel(label, digits):
    label.config(text=f"Dígitos decodificados: {digits}")

def loadDigits():
    fs, data = loadAudioFile()
    if data is not None:
        playSignal(data)
        plotTimeDomain(data, fs)
        plotSpectrogram(data, fs)
        digits = decodeSignal(fs, data)
        updateDecodedLabel(decodedLabel, digits)

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
loadBtn = ttk.Button(root, text="Cargar señal de archivo", command=loadDigits)
loadBtn.grid(row=1, column=0, pady=10)

# Etiqueta para mostrar dígitos decodificados
decodedLabel = ttk.Label(root, text="Dígitos decodificados: ")
decodedLabel.grid(row=2, column=0, pady=10)

root.mainloop()
import numpy as np
from scipy.io import wavfile
from playsound3 import playsound

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

# Función para obtener la señal de un dígito (reutilizada)
def getDigitSound(digit):
    row, col = keypad[digit]
    y1 = np.sin(2 * np.pi * fr[row] * t)
    y2 = np.sin(2 * np.pi * fc[col] * t)
    signal = (y1 + y2) / 2
    return signal * 32767

# Función para generar una pausa
def getPause(duration=0.05):  # Pausa de 0.05 segundos por defecto
    return np.zeros(int(Fs * duration))

# Generar un único archivo WAV con todos los dígitos
def generateCombinedWav():
    digits = ['2', '4', '6', '8', '2', '4', '6', '8', '2', '4', '6']  # 11 dígitos
    combined_signal = np.array([])

    for digit in digits:
        signal = getDigitSound(digit)
        combined_signal = np.concatenate((combined_signal, signal, getPause()))
    
    # Asegurar que la señal final no termine con una pausa
    combined_signal = combined_signal[:-int(Fs * 0.05)] if len(combined_signal) > int(Fs * 0.05) else combined_signal

    # Guardar el archivo WAV
    filename = 'input.wav'
    wavfile.write(filename, Fs, combined_signal.astype(np.int16))
    print(f"Guardado {filename} con duración total de {len(combined_signal) / Fs:.2f} segundos")

    # Opcional: reproducir el archivo para verificar
    # playsound(filename)

# Ejecutar la generación
if __name__ == "__main__":
    generateCombinedWav()
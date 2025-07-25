import numpy as np
from scipy.io import wavfile
from playsound3 import playsound

# Constantes
T = 0.25  # Duración de cada tono en segundos
Fs = 32768  # Frecuencia de muestreo
INTERDIGIT_PAUSE = 0.05  # Pausa de 50 ms según Q.24
t = np.linspace(0, T, int(Fs * T), endpoint=False)  # Vector de tiempo
fr = np.array([697, 770, 852, 941])  # Frecuencias de filas
fc = np.array([1209, 1336, 1477])  # Frecuencias de columnas
keypad = {
    '1': (0, 0), '2': (0, 1), '3': (0, 2),
    '4': (1, 0), '5': (1, 1), '6': (1, 2),
    '7': (2, 0), '8': (2, 1), '9': (2, 2),
    '*': (3, 0), '0': (3, 1), '#': (3, 2)
}

def getDigitSound(digit):
    row, col = keypad[digit]
    y1 = np.sin(2 * np.pi * fr[row] * t)
    y2 = np.sin(2 * np.pi * fc[col] * t)
    signal = (y1 + y2) / 2
    return signal * 32767

def getPause(duration=INTERDIGIT_PAUSE):
    return np.zeros(int(Fs * duration))

def generateCombinedWav(digits_input):
    combined_signal = np.array([])
    digits = list(digits_input)[:11]  # Limitar a 11 dígitos

    for digit in digits:
        if digit not in keypad:
            print(f"Advertencia: '{digit}' no es un dígito válido, omitido.")
            continue
        signal = getDigitSound(digit)
        combined_signal = np.concatenate((combined_signal, signal, getPause()))

    # Asegurar que la señal termine con la última pausa
    if len(combined_signal) > 0:
        combined_signal = np.concatenate((combined_signal, getPause()))

    # Guardar el archivo WAV
    filename = 'input.wav'
    wavfile.write(filename, Fs, combined_signal.astype(np.int16))
    print(f"Guardado {filename} con duración total de {len(combined_signal) / Fs:.2f} segundos")
    print(f"Número generado: {''.join(digits)}")

    # Opcional: reproducir el archivo para verificar
    # playsound(filename)

if __name__ == "__main__":
    number1 = "94518238239"
    number2 = "13571357135"
    generateCombinedWav(number1)
    # generateCombinedWav(number2)
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import datetime

# Load model
model = whisper.load_model("base")  # small bhi use kar sakti ho

fs = 16000
seconds = 5

print("Speak now...")

audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

# Convert float audio to int16 for wav
audio_int16 = np.int16(audio * 32767)
wav.write("temp.wav", fs, audio_int16)

# Transcribe audio
result = model.transcribe("temp.wav")

text = result["text"]
language = result["language"]

print("Detected Language:", language)
print("You said:", text)

# Store in file
with open("speech_log.txt", "a", encoding="utf-8") as f:
    f.write(f"{datetime.datetime.now()} | {language} | {text}\n")

    
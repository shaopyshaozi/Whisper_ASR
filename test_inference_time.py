import whisper
import torch
import time
import soundfile as sf
import numpy as np

model_name = "turbo"  # tiny, base, small, medium, large, turbo
device = "cpu" # cpu or cuda
print(device, model_name)

wav, sr = sf.read("audio.wav", always_2d=True)   # [T, C]
wav = wav.T.astype(np.float32)                  # [C, T]
x = torch.from_numpy(wav).unsqueeze(0).to(device)  # [1, C, T]

model = whisper.load_model(model_name, device=device)

print("Benchmarking...")
num_runs = 20

start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        result = model.transcribe("audio.wav")

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / num_runs
audio_length = x.shape[-1] / sr
rtf = avg_time / audio_length
print("\n===== RESULT =====")
print(f"model: {model_name}, device: {device}")
print(f"Average inference time: {avg_time:.4f} sec")
print(f"Audio length: {audio_length:.2f} sec")
print(f"RTF: {rtf:.3f}")
print("Faster than real-time" if rtf < 1 else "Slower than real-time")

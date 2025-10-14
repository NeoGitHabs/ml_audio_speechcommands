from fastapi import FastAPI, UploadFile, File, HTTPException
from numpy.conftest import dtype
from torchaudio import transforms
import torch.nn.functional as F
import torch.optim as optim
import soundfile as sf
import torch.nn as nn
import uvicorn
import torch
import io


class CheckAudio(nn.Module):
    def __init__(self,):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 35)
        )
    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio

labels = torch.load('label.pth')
index_to_labels = {ind:lab for ind, lab in enumerate(labels)}

model = CheckAudio()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('audio_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)

max_len = 100
def change_audio(waveform, sample_rate):
    if sample_rate != 16000:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        new_sr(torch.tensor(waveform))
    spec = transform(waveform).squeeze(0)
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))
    return spec

app = FastAPI()
@app.post('/predict/')
async def predict_audio(file:UploadFile = File(...,)):
    try:
        audio = await file.read()
        if not audio:
            raise HTTPExeption(status_code=400, detail='File Not Found')

    except Exception as e:
        raise HTTPExeption(status_code=500, detail=str(e))
    waveform, sample_rate = sf.read(io.BytesIO(audio), dtype='float32')
    waveform = torch.tensor(waveform).T
    spectogramma = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

    with torch.no_grad():
        y_prediction = model(spectogramma)
        prediction_ind = torch.argmax(y_prediction, dim=1).item()
        class_name = index_to_labels[prediction_ind]
        return {f'Class number: {prediction_ind}', f'Class name: {class_name}'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

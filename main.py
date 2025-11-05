import streamlit as st
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
import torch
import tempfile
import soundfile as sf
import os

# Классы Speech Commands (35 команд)
classes = ['bed', 'four', 'happy', 'nine', 'zero', 'go', 'eight', 'down', 'house', 'bird',
           'off', 'marvin', 'on', 'up', 'sheila', 'visual', 'follow', 'cat', 'yes', 'tree',
           'learn', 'six', 'no', 'left', 'one', 'two', 'stop', 'seven', 'backward', 'three',
           'dog', 'wow', 'five', 'forward', 'right']

# Параметры из notebook
SAMPLE_RATE = 16000
N_MELS = 64
MAX_LEN = 100

transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)


class CheckAudio(nn.Module):
    def __init__(self):
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
            nn.Linear(128, 35)  # 35 классов
        )

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio


# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckAudio()
model.load_state_dict(torch.load('audioCpeechCommands_model.pth', map_location=device))
model.to(device)
model.eval()

st.title('Speech Commands Classifier')
st.text('Загрузите аудио команду, и модель попробует её распознать.')

audio_file = st.file_uploader('Выберите аудио', type=['wav', 'mp3', 'flac', 'ogg'])

if not audio_file:
    st.info('Загрузите аудио')
else:
    st.audio(audio_file)

    if st.button('Распознать'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name

            # Загрузка через soundfile
            waveform, sr = sf.read(tmp_path)
            waveform = torch.from_numpy(waveform).float()

            # Преобразование в формат [channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T

            os.unlink(tmp_path)

            # Ресемплинг если нужно
            if sr != SAMPLE_RATE:
                resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            # Спектрограмма + паддинг
            spec = transform(waveform).squeeze(0)

            if spec.shape[1] > MAX_LEN:
                spec = spec[:, :MAX_LEN]
            elif spec.shape[1] < MAX_LEN:
                count_len = MAX_LEN - spec.shape[1]
                spec = F.pad(spec, (0, count_len))

            spec = spec.unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(spec)
                prediction = y_prediction.argmax(dim=1).item()

            st.success(f'Модель думает, что это команда: {classes[prediction]}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')




# app = FastAPI()
#
# @app.post('/predict')
# async def check_image(file:UploadFile = File(...)):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(status_code=400, detail='File not Found')
#
#         img = Image.open(io.BytesIO(data))
#         img_tensor = transform(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             prediction = model(img_tensor)
#             result = prediction.argmax(dim=1).item()
#             return {f'class': classes[result]}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'{e}')
#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
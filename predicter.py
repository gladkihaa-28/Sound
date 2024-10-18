import os
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm

# Путь к файлам в папке test
test_directory = 'test'

# Загрузка обученной модели
best_model_path = 'best_model.pth'  # Путь к сохраненной модели
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=3)
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Перевод модели в режим инференса

# Загрузка процессора для аудио
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Функция для обработки аудио файла
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)  # Загрузка и ресемплинг
    audio_input = processor(audio, return_tensors='pt', sampling_rate=16000,
                            padding=True, truncation=True, max_length=16000)
    return audio_input.input_values[0]

# Получение списка всех файлов в тестовой папке
test_files = [os.path.join(test_directory, f) for f in os.listdir(test_directory) if f.endswith('.mp3')]

# Подготовка данных для предсказаний
test_processed = np.array([preprocess_audio(file) for file in tqdm(test_files, desc="Processing test files")])
X_test_tensor = torch.tensor(test_processed)

# Перенос на устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_test_tensor = X_test_tensor.to(device)

# Предсказания
with torch.no_grad():
    y_pred = model(X_test_tensor).logits
    y_pred_classes = np.argmax(y_pred.cpu().numpy(), axis=1)  # Прогнозируемые метки

# Формирование DataFrame для записи в CSV
results = pd.DataFrame({
    'filename': [os.path.basename(file) for file in test_files],
    'label': y_pred_classes
})

# Сохранение результатов в CSV
output_csv = 'predictions.csv'
results.to_csv(output_csv, index=False)

print(f"Результаты сохранены в файл: {output_csv}")

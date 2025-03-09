import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Отключаем oneDNN для устранения сообщений TensorFlow

from collections import deque, OrderedDict
import json
import logging
import hashlib
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import pygame
from pygame import mixer
import time
import threading
import ast
from typing import Dict, Any, List, Tuple, Optional
import re
from bs4 import BeautifulSoup
from docx import Document
import openpyxl
from tinydb import TinyDB, Query
import joblib
from cryptography.fernet import Fernet
import black
import requests
import importlib.util
import sys
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import base64
from gtts import gTTS  # Добавлен импорт gTTS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Импорты TensorFlow
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam, SGD
except ImportError as e:
    logging.error(f"Ошибка импорта tensorflow.keras: {e}")
    print("Пожалуйста, установите TensorFlow: 'pip install tensorflow'")

# Импорты Fast.ai
try:
    from fastai.text.all import *
    from fastai.text.models import AWD_LSTM
except ImportError as e:
    logging.error(f"Ошибка импорта fastai: {e}")
    print("Пожалуйста, установите Fast.ai: 'pip install fastai'")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("nere_more.log"), logging.StreamHandler()],
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Класс для анализа текста с Fast.ai (регрессия)
class FastAITextAnalyzer:
    def __init__(self, csv_path="sentiment_data.csv"):
        self.csv_path = csv_path
        self.dls = None
        self.learn = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Загружаем или обучаем регрессионную модель Fast.ai с реальными данными."""
        try:
            if os.path.exists("text_regressor.pth"):
                self.learn = load_learner("text_regressor.pth")
                logging.info("Загружена сохраненная регрессионная модель Fast.ai")
            else:
                # Загружаем данные из CSV или создаем небольшой набор отзывов
                if os.path.exists(self.csv_path):
                    data = pd.read_csv(self.csv_path)
                    if 'text' not in data.columns or 'score' not in data.columns:
                        raise ValueError("CSV должен содержать колонки 'text' и 'score' (от -1 до 1)")
                else:
                    # Небольшой набор данных с отзывами и числовыми оценками
                    data = pd.DataFrame({
                        'text': [
                            "Отличный сервис, я в восторге!",
                            "Ужасное обслуживание, никогда не вернусь",
                            "Всё нормально, ничего особенного",
                            "Очень грустно от такого качества",
                            "Злюсь на вашу доставку, это кошмар",
                            "Прекрасный день благодаря вам",
                            "Так себе, могло быть лучше",
                            "Просто отвратительно",
                            "Супер, всё идеально!",
                            "Разочарован, ожидал большего"
                        ],
                        'score': [1.0, -1.0, 0.0, -0.6, -0.9, 0.8, -0.2, -0.8, 1.0, -0.4]
                    })
                    data.to_csv(self.csv_path, index=False)
                    logging.warning(f"Создан примерный файл {self.csv_path}. Замените его на свои данные.")

                # Создаем DataLoaders для регрессии
                dls = TextDataLoaders.from_df(
                    data,
                    text_col='text',
                    label_col='score',
                    valid_pct=0.2,
                    text_vocab=None,
                    is_lm=False
                )

                # Используем регрессионную модель
                self.learn = text_learner(
                    dls,
                    AWD_LSTM,
                    drop_mult=0.5,
                    metrics=[mae],  # Средняя абсолютная ошибка
                    loss_func=MSELossFlat()  # Среднеквадратичная ошибка для регрессии
                )

                # Дообучаем модель
                self.learn.fit_one_cycle(3, 1e-2)
                self.learn.export("text_regressor.pth")
                logging.info("Создана и сохранена новая регрессионная модель Fast.ai")
            self.dls = self.learn.dls
        except Exception as e:
            logging.error(f"Ошибка инициализации Fast.ai модели: {e}")
            self.learn = None

    def predict_sentiment_score(self, text: str) -> float:
        """Предсказываем числовую оценку настроения от -1 до 1."""
        if not self.learn:
            return 0.0
        try:
            pred = self.learn.predict(text)[0].item()  # Получаем числовое предсказание
            return float(pred)
        except Exception as e:
            logging.error(f"Ошибка предсказания Fast.ai: {e}")
            return 0.0

    def fine_tune(self, text: str, score: float):
        """Дообучаем модель на новом примере."""
        if not self.learn:
            return
        try:
            df = pd.DataFrame({'text': [text], 'score': [score]})
            dls = TextDataLoaders.from_df(df, text_col='text', label_col='score', valid_pct=0)
            self.learn.dls = dls
            self.learn.fine_tune(1, base_lr=1e-3)
            self.learn.export("text_regressor.pth")
            logging.info(f"Модель Fast.ai дообучена на: {text} -> {score:.2f}")
            # Добавляем данные в CSV
            with open(self.csv_path, 'a', encoding='utf-8') as f:
                f.write(f'"{text}",{score}\n')
        except Exception as e:
            logging.error(f"Ошибка дообучения Fast.ai: {e}")

class CodeEditorWindow(ctk.CTkToplevel):
    def __init__(self, parent, services):
        super().__init__(parent)
        self.title("Редактор кода")
        self.geometry("600x400")
        self.services = services
        self._init_ui()

    def _init_ui(self):
        self.code_textbox = ctk.CTkTextbox(self, width=580, height=300, fg_color="#2F3536", text_color="#FFFFFF")
        self.code_textbox.pack(padx=10, pady=10)
        ctk.CTkButton(self, text="Анализировать", command=self._analyze_code).pack(pady=5)

    def _analyze_code(self):
        code = self.code_textbox.get("1.0", "end-1c").strip()
        if code:
            purpose, location = self.services.code_optimizer.classify_code(code)
            self.code_textbox.delete("1.0", "end")
            self.code_textbox.insert("1.0", f"Код:\n{code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}")

def validate_folder_id(folder_id: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9]{20}$', folder_id))

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class AudioManager:
    def play_sound(self, text: str, filename: str) -> None:
        try:
            tts = gTTS(text, lang="ru")
            tts.save(filename)
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            if os.path.exists(filename):
                os.remove(filename)
        except (pygame.error, OSError) as e:
            logging.error(f"Ошибка воспроизведения аудио: {e}")

class Config:
    _instance = None
    _lock = threading.Lock()
    _cipher_key = Fernet.generate_key()
    _cipher = Fernet(_cipher_key)

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._load_config()
                cls._instance._validate_and_update_on_startup()
            return cls._instance

    def _load_config(self):
        self.default_config = {
            "database": "nere_more_knowledge.json",
            "yandex": {
                "keys": [{"id": "gpt_key_1", "value": "", "type": "gpt"}],
                "folder_id": ""
            },
            "ui": {"animation_speed": 0.05, "max_context": 10, "audio_enabled": True},
            "learning": {"cluster_size": 5, "feedback_weight": 0.9, "learning_rate": 0.0},
            "code_classification": {
                "purposes": ["algorithm", "UI", "data_processing", "network", "utility"],
                "locations": ["core", "GUI", "services", "knowledge"],
                "keywords": {
                    "algorithm": ["def", "for", "while", "if"],
                    "UI": ["tkinter", "ctk", "label", "button"],
                    "data_processing": ["numpy", "sklearn", "pandas"],
                    "network": ["requests", "socket"],
                    "utility": ["os", "logging", "time"]
                }
            }
        }
        config_file = "nere_more_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        self.config = self.default_config
                        self._save_config()
                        return
                    self.config = json.loads(content)
                    temp_config = self.default_config.copy()
                    temp_config.update(self.config)
                    self.config = temp_config
            else:
                self.config = self.default_config
                self._save_config()
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            self.config = self.default_config
            self._save_config()

    def _save_config(self):
        try:
            with open("nere_more_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except IOError as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")

    def _validate_and_update_on_startup(self):
        api_key = self.get_key()
        folder_id = self.get_folder_id()
        
        if not api_key or not folder_id:
            logging.info("API ключ или folder_id отсутствуют, используются значения по умолчанию")
            self.update_api_key("gpt_key_1", "AQVNzHvgRbhMqf98hCeuO8ek88XTmHFnVJ3fKcmo")
            self.update_folder_id("b1g170pkl3ihbn8bc3kd")
            return
            
        temp_gpt = YandexGPT(api_key, folder_id)
        available, status = temp_gpt.check_availability()
        
        if not available:
            logging.warning(f"Сохраненные данные недействительны: {status}. Установка значений по умолчанию")
            self.update_api_key("gpt_key_1", "AQVNzHvgRbhMqf98hCeuO8ek88XTmHFnVJ3fKcmo")
            self.update_folder_id("b1g170pkl3ihbn8bc3kd")

    @property
    def data(self) -> Dict[str, Any]:
        return self.config

    def get_key(self) -> str:
        for key in self.config["yandex"]["keys"]:
            if key["type"] == "gpt":
                try:
                    return self._cipher.decrypt(key["value"].encode()).decode()
                except Exception:
                    return key["value"]
        return ""

    def update_api_key(self, key_id: str, value: str) -> bool:
        temp_gpt = YandexGPT(value, self.get_folder_id())
        available, status = temp_gpt.check_availability()
        
        if not available:
            logging.error(f"Новый API ключ недействителен: {status}")
            return False
            
        encrypted_value = self._cipher.encrypt(value.encode()).decode()
        for key in self.config["yandex"]["keys"]:
            if key["id"] == key_id:
                key["value"] = encrypted_value
                self._save_config()
                return True
        self.config["yandex"]["keys"].append({"id": key_id, "value": encrypted_value, "type": "gpt"})
        self._save_config()
        return True

    def get_folder_id(self) -> str:
        return self.config["yandex"].get("folder_id", "")

    def update_folder_id(self, folder_id: str) -> bool:
        if not validate_folder_id(folder_id):
            logging.error("Недействительный folder_id")
            return False
            
        api_key = self.get_key()
        if api_key:
            temp_gpt = YandexGPT(api_key, folder_id)
            available, status = temp_gpt.check_availability()
            if not available:
                logging.error(f"folder_id недействителен с текущим ключом: {status}")
                return False
                
        self.config["yandex"]["folder_id"] = folder_id
        self._save_config()
        return True

class CodeOptimizationModule:
    def __init__(self, config: Config):
        self.config = config.data["code_classification"]

    def classify_code(self, code: str) -> Tuple[str, str]:
        purpose_score = {p: 0 for p in self.config["purposes"]}
        tokens = re.findall(r'\w+', code.lower())
        
        for purpose, keywords in self.config["keywords"].items():
            for keyword in keywords:
                if keyword in tokens:
                    purpose_score[purpose] += 1
        
        purpose = max(purpose_score, key=purpose_score.get, default="utility")
        
        if "ctk" in code or "tkinter" in code:
            location = "GUI"
        elif "requests" in code or "socket" in code:
            location = "services"
        elif "numpy" in code or "sklearn" in code:
            location = "knowledge"
        else:
            location = "core"
        
        return purpose, location

    def detect_errors(self, code: str) -> List[str]:
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Синтаксическая ошибка: {str(e)}")
        return errors if errors else ["Ошибок не обнаружено"]

    def analyze_structure(self, code: str) -> Dict[str, List[str]]:
        tree = ast.parse(code)
        structure = {"functions": [], "classes": []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append(node.name)
        
        return structure

    def suggest_structure(self, code: str, errors: List[str]) -> str:
        suggestions = []
        try:
            tree = ast.parse(code, mode='exec', type_comments=True)
        except:
            suggestions.append("Исправьте синтаксическую ошибку для дальнейшего анализа.")
            return "\n".join(suggestions)
        
        for error in errors:
            if "syntax" in error.lower():
                suggestions.append("Исправьте синтаксическую ошибку (например, скобки, отступы).")
        
        return "\n".join(suggestions) if suggestions else "Структура корректна."

    def duplicate_structure(self, code: str) -> str:
        tree = ast.parse(code)
        duplicated_code = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                duplicated_code.append(ast.unparse(node))
        
        return "\n\n".join(duplicated_code) if duplicated_code else code

    def suggest_integration_points(self, code: str, location: str) -> List[Tuple[str, str]]:
        integration_points = []
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                integration_points.append((node.name, f"Интеграция функции {node.name} в {location}"))
            elif isinstance(node, ast.ClassDef):
                integration_points.append((node.name, f"Интеграция класса {node.name} в {location}"))
        
        return integration_points

class YandexGPT:
    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.model = "yandexgpt-lite"
        self.temperature = 0.6
        self.max_tokens = 2000
        self.available = False
        self.status = "Не проверено"
        self._validate_credentials()

    def _validate_credentials(self):
        if not self.api_key or len(self.api_key.strip()) < 10:
            self.status = "Ошибка: API-ключ пустой или слишком короткий"
            return
        if not validate_folder_id(self.folder_id):
            self.status = "Ошибка: folder_id должен быть 20 символов (буквы/цифры)"
            return
        self.check_availability()

    def check_availability(self) -> Tuple[bool, str]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}"
            }
            payload = {
                "modelUri": f"gpt://{self.folder_id}/{self.model}",
                "completionOptions": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature
                },
                "messages": [{"role": "user", "text": "Test"}]
            }
            response = requests.post(self.url, headers=headers, json=payload)
            self.available = response.status_code == 200
            self.status = "Доступно" if self.available else f"Ошибка: {response.status_code}"
            return self.available, self.status
        except Exception as e:
            self.available = False
            self.status = f"Ошибка сети: {str(e)}"
            return self.available, self.status

    def invoke(self, json_payload: Dict[str, Any]) -> str:
        if not self.available:
            return f"API отключен: {self.status}"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}"
            }
            response = requests.post(self.url, headers=headers, json=json_payload)
            response.raise_for_status()
            result = response.json()
            return result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "No data")
        except requests.RequestException as e:
            logging.error(f"Ошибка запроса к Yandex GPT: {e}")
            return f"Ошибка сети: {str(e)}"

class KnowledgeBase:
    def __init__(self, services: 'YandexAIServices'):
        self.config = Config().data
        self.services = services
        self.db = TinyDB(self.config["database"])
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.kmeans: Optional[MiniBatchKMeans] = None
        self._lock = threading.Lock()
        self.embeddings_cache = OrderedDict(maxlen=1000)
        self.query_cache = OrderedDict(maxlen=1000)
        self.vectorizer_fitted = False
        self.experience_level = 0
        self.learning_rate = 0.0
        self.context_history = deque(maxlen=50)
        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists("vectorizer.pkl") and os.path.exists("kmeans.pkl"):
                self.vectorizer = joblib.load("vectorizer.pkl")
                self.kmeans = joblib.load("kmeans.pkl")
                self.vectorizer_fitted = True
            else:
                docs = [entry['response'] for entry in self.db.all() if 'response' in entry]
                if docs:
                    self.vectorizer.fit(docs)
                    self.vectorizer_fitted = True
                    n_samples = len(docs)
                    n_clusters = min(self.config["learning"]["cluster_size"], n_samples)
                    self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
                    embeddings = []
                    for entry in self.db.all():
                        if 'embeddings' in entry:
                            embeddings_bytes = base64.b64decode(entry['embeddings'])
                            embeddings.append(np.frombuffer(embeddings_bytes, dtype=np.float16))
                    if embeddings:
                        self.kmeans.fit(np.stack(embeddings))
                    self._save_state()
            self.experience_level = len(self.db.all())
            self.learning_rate = self.config["learning"]["learning_rate"]
        except Exception as e:
            logging.error(f"Ошибка загрузки состояния: {e}")
            self.vectorizer_fitted = False

    def _save_state(self):
        if self.vectorizer_fitted:
            try:
                joblib.dump(self.vectorizer, "vectorizer.pkl")
                joblib.dump(self.kmeans, "kmeans.pkl")
            except Exception as e:
                logging.error(f"Ошибка сохранения моделей: {e}")

    def _ensure_vectorizer_fitted(self, text: str):
        if not self.vectorizer_fitted:
            docs = [entry['response'] for entry in self.db.all() if 'response' in entry]
            if not docs:
                docs = [text]
            self.vectorizer.fit(docs)
            self.vectorizer_fitted = True
            n_clusters = min(self.config["learning"]["cluster_size"], len(docs))
            self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
            self._save_state()

    def update_experience(self) -> float:
        self.experience_level += 1
        fib_value = fibonacci(min(self.experience_level, 20))
        self.learning_rate = min(100.0, fib_value * 0.5)
        self.config["learning"]["learning_rate"] = self.learning_rate
        Config()._save_config()
        return self.learning_rate

    def save(self, query: str, response: str, context: str = "", feedback: float = 0.0):
        with self._lock:
            if len(response.strip()) < 5:
                return
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.query_cache[query_hash] = response
            self._ensure_vectorizer_fitted(response)
            query_vec = self.vectorizer.transform([query]).toarray().astype(np.float16)[0]
            embeddings = self.vectorizer.transform([response]).toarray().astype(np.float16)[0]
            embeddings_base64 = base64.b64encode(embeddings.tobytes()).decode('utf-8')
            entry = {
                'id': str(uuid.uuid4()),
                'query': query,
                'response': response,
                'context': context,
                'embeddings': embeddings_base64,
                'feedback': feedback,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'learning_rate': self.update_experience()
            }
            self.db.insert(entry)
            self.context_history.append(f"Вопрос: {query}\nОтвет: {response}")

    def get_similar(self, query: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        self._ensure_vectorizer_fitted(query)
        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float16)[0]
        all_data = []
        for entry in self.db.all():
            if 'embeddings' in entry:
                embeddings_bytes = base64.b64decode(entry['embeddings'])
                embeddings_array = np.frombuffer(embeddings_bytes, dtype=np.float16)
                all_data.append((entry['query'], entry['response'], embeddings_array))
        if not all_data:
            return []
        embeddings = np.stack([item[2] for item in all_data])
        similarities = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
        results = sorted(zip(all_data, similarities), key=lambda x: x[1], reverse=True)[:top_n]
        return [(q, r, float(s)) for (q, r, _), s in results]

    def save_web_content(self, url: str, query: str) -> bool:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            if text:
                self.save(query, text, context=f"Извлечено с сайта: {url}")
                return True
            return False
        except Exception as e:
            logging.error(f"Ошибка извлечения с {url}: {e}")
            return False

    def build_context(self, query: str) -> str:
        similar = self.get_similar(query, top_n=3)
        context = "\n\n".join([f"Ранее: {q}\nОтвет: {r} (сходство: {s:.2f})" for q, r, s in similar])
        if not context:
            context = "У меня пока мало опыта по этому вопросу, но я постараюсь ответить на основе доступных данных."
        history = "\n".join(list(self.context_history)[-5:])
        return f"История взаимодействия:\n{history}\n\nПрошлый опыт:\n{context}"

    def _clear_cache(self, text_id: str = None):
        with self._lock:
            if text_id:
                self.db.remove(Query().query.matches(f".*ID: {text_id}.*"))
            else:
                self.db.truncate()
                self.embeddings_cache.clear()
                self.query_cache.clear()
                self.context_history.clear()
                self.vectorizer_fitted = False
                self.kmeans = None
                self.experience_level = 0
                self.learning_rate = 0.0
            self._save_state()

class DataModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        logging.info("Инициализация DataModelTrainer")

    def generate_synthetic_data(self, n_samples=1000):
        X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Сгенерированы синтетические данные")

    def build_model(self, optimizer_name="Adam", learning_rate=0.001):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        if optimizer_name == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        logging.info(f"Модель создана с оптимизатором {optimizer_name} и learning_rate={learning_rate}")

    def train_model(self, epochs=50, batch_size=32):
        if self.model is None or self.X_train is None:
            raise ValueError("Модель или данные не инициализированы")
        
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.X_test, self.y_test), verbose=1)
        logging.info(f"Модель обучена: {epochs} эпох, размер батча {batch_size}")

    def visualize_results(self, output_path="training_results.png"):
        if self.history is None:
            logging.error("История обучения отсутствует")
            return

        history_df = pd.DataFrame(self.history.history)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.lineplot(data=history_df[['loss', 'val_loss']])
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Графики сохранены в {output_path}")

class InteractiveBehavior:
    def __init__(self, gui_interface):
        self.gui = gui_interface
        self.last_interaction = time.time()
        self.user_mood = 0.0
        self.greetings = ["Привет!", "Здорово!", "Как дела?"]
        self.questions = ["Чем занимаешься?", "Почему молчишь?", "Как у вас дела?", "Что нового?"]
        self.suggestions = [
            "Может, обучим модель на новых данных?",
            "Давай проанализируем какой-нибудь код?",
            "Хочешь узнать что-то интересное?",
            "Как насчет визуализации данных?"
        ]
        self.is_running = False
        self.thread = None
        self.text_analyzer = FastAITextAnalyzer()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._interaction_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _interaction_loop(self):
        while self.is_running:
            time_since_last = time.time() - self.last_interaction
            if time_since_last > 30:
                self._interact_with_user()
                self.last_interaction = time.time()
            time.sleep(5)

    def _interact_with_user(self):
        last_input = self.gui.input_entry.get().strip()
        if last_input:
            score = self.text_analyzer.predict_sentiment_score(last_input)
            self.user_mood = score
            logging.info(f"Fast.ai анализ: {last_input} -> оценка: {score:.2f}")
            self.text_analyzer.fine_tune(last_input, score)

        if time.time() - self.last_interaction > 60:
            question = random.choice(self.questions)
            self.gui.display_response(question)
        else:
            if self.user_mood > 0.5:
                greeting = random.choice(self.greetings) + " Ты выглядишь счастливым!"
            elif self.user_mood < -0.5:
                greeting = random.choice(self.greetings) + " Не грусти, давай что-нибудь сделаем!"
            else:
                greeting = random.choice(self.greetings)
            suggestion = random.choice(self.suggestions)
            self.gui.display_response(f"{greeting}\n{suggestion}")

    def update_last_interaction(self):
        self.last_interaction = time.time()

class YandexAIServices:
    def __init__(self, gui_parent=None):
        self.config = Config()
        self.gui_parent = gui_parent
        self._request_credentials_if_needed()
        self.knowledge = KnowledgeBase(self)
        self.gpt = YandexGPT(self.config.get_key(), self.config.get_folder_id())
        self.code_optimizer = CodeOptimizationModule(self.config)
        self.data_trainer = DataModelTrainer()
        self.text_analyzer = FastAITextAnalyzer()
        
        available, status = self.gpt.check_availability()
        if not available:
            logging.warning(f"Инициализация с проблемой API: {status}")
            if gui_parent:
                gui_parent.status_label.configure(text=f"Ошибка API: {status}")

    def _request_credentials_if_needed(self):
        if not validate_folder_id(self.config.get_folder_id()):
            folder_id = ctk.CTkInputDialog(text="Введите folder_id (20 символов):", title="Folder ID").get_input()
            if folder_id and self.config.update_folder_id(folder_id):
                logging.info(f"Обновлен folder_id: {folder_id}")
        if not self.config.get_key():
            api_key = ctk.CTkInputDialog(text="Введите API-ключ:", title="API Key").get_input()
            if api_key and self.config.update_api_key("gpt_key_1", api_key):
                logging.info("API ключ обновлен")

    def check_api_key(self) -> Tuple[bool, str]:
        return self.gpt.check_availability()

    def train_and_visualize(self, epochs=50, batch_size=32, optimizer_name="Adam", learning_rate=0.001):
        try:
            self.data_trainer.generate_synthetic_data()
            self.data_trainer.build_model(optimizer_name=optimizer_name, learning_rate=learning_rate)
            self.data_trainer.train_model(epochs=epochs, batch_size=batch_size)
            self.data_trainer.visualize_results()
            return f"Модель обучена! Графики сохранены в training_results.png\nОпыт ИИ: {self.knowledge.learning_rate:.1f}%"
        except Exception as e:
            logging.error(f"Ошибка при обучении модели: {e}")
            return f"Ошибка при обучении: {str(e)}"

    def suggest_action_algorithm(self, query: str, user_emotion: Optional[float] = None) -> str:
        if user_emotion is None:
            user_emotion = self.text_analyzer.predict_sentiment_score(query)
            logging.info(f"Fast.ai анализ запроса: {query} -> оценка: {user_emotion:.2f}")
            self.text_analyzer.fine_tune(query, user_emotion)

        keywords = re.findall(r'\w+', query.lower())
        main_focus = max(keywords, key=lambda w: len(w), default="запрос")

        similar_entries = self.knowledge.get_similar(query, top_n=5)
        memory_context = "\n".join([f"[{s:.2f}] {r}" for _, r, s in similar_entries]) if similar_entries else "Нет схожих данных"

        if user_emotion > 0.5:
            tone = "радостный"
            suggestion = "Продолжайте в том же духе, я с радостью помогу!"
        elif user_emotion < -0.5:
            tone = "поддерживающий"
            suggestion = "Не переживайте, я здесь, чтобы помочь!"
        else:
            tone = "нейтральный"
            suggestion = "Давайте разберем ваш запрос вместе."

        built_context = self.knowledge.build_context(query)
        prompt = {
            "modelUri": f"gpt://{self.config.get_folder_id()}/{self.gpt.model}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.5,
                "maxTokens": 2000
            },
            "messages": [
                {
                    "role": "system",
                    "text": f"Ты ассистент с уровнем опыта {self.knowledge.learning_rate:.1f}%. Используй накопленные знания и адаптируйся к эмоциональному состоянию пользователя (тон: {tone})."
                },
                {
                    "role": "user",
                    "text": f"Контекст:\n{memory_context}\n\nЗапрос: {query}\n\nОсновной фокус: {main_focus}"
                }
            ]
        }
        response = self.gpt.invoke(prompt)
        resp_score = self.text_analyzer.predict_sentiment_score(response)
        logging.info(f"Fast.ai анализ ответа: {response[:50]}... -> оценка: {resp_score:.2f}")
        self.knowledge.save(query, response, context=f"Эмоциональный тон: {tone}, Фокус: {main_focus}")
        return (
            f"Основной фокус: {main_focus}\n"
            f"Эмоциональный тон: {tone}\n"
            f"Оценка настроения запроса: {user_emotion:.2f}\n"
            f"Предложение: {suggestion}\n"
            f"Ответ: {response}\n"
            f"Оценка настроения ответа: {resp_score:.2f}\n"
            f"[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]"
        )

    def generate_response(self, query: str, context: str = "") -> str:
        if not query:
            return "Ошибка: Запрос пуст"
        
        if "код" not in query.lower() and "code" not in query.lower() and not re.findall(r'https?://\S+', query):
            return self.suggest_action_algorithm(query)
        
        urls = re.findall(r'https?://\S+', query)
        if urls:
            success = self.knowledge.save_web_content(urls[0], query)
            return f"Сохранено с {urls[0]}\n[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]" if success else f"Ошибка с {urls[0]}"
        
        if "код" in query.lower() or "code" in query.lower():
            try:
                formatted_code = black.format_str(query, mode=black.FileMode())
                purpose, location = self.code_optimizer.classify_code(query)
                errors = self.code_optimizer.detect_errors(query)
                response = f"Отформатированный код:\n{formatted_code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}\n\nОшибки:\n" + "\n".join(errors)
                self.knowledge.save(query, response)
                return f"{response}\n[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]"
            except Exception as e:
                return f"Ошибка обработки кода: {e}"
        
        built_context = self.knowledge.build_context(query)
        prompt = {
            "modelUri": f"gpt://{self.config.get_folder_id()}/{self.gpt.model}",
            "completionOptions": {
                "stream": False,
                "temperature": max(0.3, 0.6 - (self.knowledge.learning_rate / 200)),
                "maxTokens": 2000
            },
            "messages": [
                {
                    "role": "system",
                    "text": f"Ты ассистент, который учится на основе опыта (текущий уровень: {self.knowledge.learning_rate:.1f}%). Используй накопленные знания для логически последовательных ответов."
                },
                {
                    "role": "user",
                    "text": f"Контекст:\n{built_context}\n\nТекущий запрос: {query}"
                }
            ]
        }
        response = self.gpt.invoke(prompt)
        resp_score = self.text_analyzer.predict_sentiment_score(response)
        logging.info(f"Fast.ai анализ ответа: {response[:50]}... -> оценка: {resp_score:.2f}")
        self.knowledge.save(query, response, context=built_context)
        return f"{response}\n\nОценка настроения ответа: {resp_score:.2f}\n[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]"

class CodePasteWindow(ctk.CTkToplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Вставка кода")
        self.geometry("400x300")
        self.callback = callback
        self._init_ui()

    def _init_ui(self):
        self.code_entry = ctk.CTkTextbox(self, width=380, height=200, fg_color="#1C2526", text_color="#FFFFFF", font=("Courier", 12))
        self.code_entry.pack(padx=10, pady=10, fill="both", expand=True)
        self.code_entry.insert("1.0", "# Вставьте код здесь\n")

        button_frame = ctk.CTkFrame(self, fg_color="#2F3536")
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="Вставить", command=self._paste_code, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Увеличить", command=self._enlarge_window, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Уменьшить", command=self._shrink_window, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Отмена", command=self.destroy, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)

    def _paste_code(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if code:
            self.callback(code)
        self.destroy()

    def _enlarge_window(self):
        current_width, current_height = map(int, self.geometry().split('x')[0:2])
        self.geometry(f"{current_width + 100}x{current_height + 100}")

    def _shrink_window(self):
        current_width, current_height = map(int, self.geometry().split('x')[0:2])
        if current_width > 200 and current_height > 200:
            self.geometry(f"{current_width - 100}x{current_height - 100}")

class NereMoreInterface(ctk.CTk):
    def __init__(self):
        logging.info("Шаг 0: Начало инициализации NereMoreInterface")
        try:
            super().__init__()
            self.title("Nere More")
            self.geometry("600x450")
            self.configure(fg_color="#1C2526")
            self.initialized = False

            self.audio = AudioManager()
            self.services = YandexAIServices(self)
            self.config = Config()
            self.context = deque(maxlen=self.config.data["ui"]["max_context"] * 2)
            self.interactive_behavior = InteractiveBehavior(self)

            self.logo_label = ctk.CTkLabel(self, text="Nere More", font=("Arial", 20, "bold"), text_color="#FFFFFF")
            self.logo_label.pack(pady=10)

            self.input_frame = ctk.CTkFrame(self, fg_color="#2F3536", corner_radius=10)
            self.input_frame.pack(fill="x", padx=10, pady=5)
            self.input_entry = ctk.CTkEntry(self.input_frame, width=350, height=40, font=("Arial", 14),
                                            placeholder_text="Введите запрос...", fg_color="#1C2526", text_color="#FFFFFF")
            self.input_entry.pack(side="left", padx=10, pady=5)
            self.input_entry.bind("<Return>", lambda e: self.process_input_with_interaction())

            buttons = [
                ("📋", lambda: CodePasteWindow(self, self._paste_text_callback), "Вставить"),
                ("🧲", self._magnet_search, "Поиск"),
                ("🔍", self.process_input, "Обработать"),
                ("🌟", self.show_skills, "Умения"),
                ("⚙️", lambda: APISettingsWindow(self, self.config), "Настройки"),
                ("🔑", lambda: APIKeyCheckWindow(self, self.services, self.config), "Проверка ключа"),
                ("💻", lambda: CodeEditorWindow(self, self.services), "Редактор кода"),
                ("📊", self.train_model, "Обучить модель"),
            ]
            for text, cmd, hover in buttons:
                btn = ctk.CTkButton(self.input_frame, text=text, width=40, height=40, fg_color="#1C2526", hover_color="#4A4A4A",
                                   text_color="#FFFFFF", command=cmd)
                btn.pack(side="left", padx=5)

            self.results_text = ctk.CTkTextbox(self, width=580, height=300, fg_color="#2F3536", text_color="#FFFFFF")
            self.results_text.pack(padx=10, pady=5)

            self.button_frame = ctk.CTkFrame(self, fg_color="#2F3536")
            self.button_frame.pack(fill="x", padx=10, pady=5)
            compact_buttons = [
                ("📋", lambda: CodePasteWindow(self, self._paste_text_callback), "Вставить"),
                ("🧲", self._magnet_search, "Поиск"),
                ("🔍", self.process_input, "Обработать"),
                ("🌟", self.show_skills, "Умения"),
                ("⚙️", lambda: APISettingsWindow(self, self.config), "Настройки"),
                ("🔑", lambda: APIKeyCheckWindow(self, self.services, self.config), "Проверка"),
                ("💻", lambda: CodeEditorWindow(self, self.services), "Редактор"),
                ("📊", self.train_model, "Обучить"),
            ]
            for text, cmd, hover in compact_buttons:
                btn = ctk.CTkButton(self.button_frame, text=text, width=30, height=30, fg_color="#1C2526", hover_color="#4A4A4A",
                                   text_color="#FFFFFF", command=cmd)
                btn.pack(side="left", padx=2)

            self.status_label = ctk.CTkLabel(self, text="Инициализация...", font=("Arial", 10), text_color="#FFFFFF")
            self.status_label.pack(side="bottom", pady=2)

            available, status = self.services.check_api_key()
            self.status_label.configure(text=f"Статус API: {status}")

            self.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.initialized = True
            self.status_label.configure(text=f"Готов [Опыт ИИ: {self.services.knowledge.learning_rate:.1f}%]")
            self.interactive_behavior.start()
        except Exception as e:
            logging.error(f"Критическая ошибка инициализации: {e}", exc_info=True)
            messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение: {e}")
            self.destroy()

    def _on_closing(self):
        logging.info("Закрытие приложения")
        self.interactive_behavior.stop()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self.services.knowledge.db.close()
        self.destroy()

    def display_response(self, text: str):
        self.results_text.delete("1.0", "end")
        try:
            json_response = json.loads(text)
            self.results_text.insert("1.0", json.dumps(json_response, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            self.results_text.insert("1.0", text)
        self.status_label.configure(text=f"Готов [Опыт ИИ: {self.services.knowledge.learning_rate:.1f}%]")

    def process_input(self):
        query = self.input_entry.get().strip()
        if not query:
            return
        self.input_entry.delete(0, "end")
        if query.lower().startswith("clear cache"):
            text_id = query.split("ID:")[-1].strip() if "ID:" in query else None
            self.services.knowledge._clear_cache(text_id)
            self.display_response(f"Кэш {'для ID: ' + text_id if text_id else 'полностью'} очищен")
        else:
            response = self.services.generate_response(query, self._get_context())
            self.display_response(response)
            self.context.append({"role": "user", "content": query})
            self.context.append({"role": "assistant", "content": response})

    def process_input_with_interaction(self):
        self.interactive_behavior.update_last_interaction()
        self.process_input()

    def train_model(self):
        response = self.services.train_and_visualize(epochs=50, batch_size=32, optimizer_name="Adam", learning_rate=0.001)
        self.display_response(response)

    def _paste_text_callback(self, content):
        if "def" in content or "class" in content:
            purpose, location = self.services.code_optimizer.classify_code(content)
            errors = self.services.code_optimizer.detect_errors(content)
            formatted_code = black.format_str(content, mode=black.FileMode()) if not errors else content
            response = f"Код вставлен:\n{formatted_code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}\n\nОшибки:\n" + "\n".join(errors)
            self.services.knowledge.save(f"Inserted code (ID: {uuid.uuid4().hex[:8]})", formatted_code)
            self.display_response(response)
        else:
            self.services.knowledge.save(f"Inserted text (ID: {uuid.uuid4().hex[:8]})", content)
            self.display_response(f"Вставлено: {content[:100]}...")

    def _magnet_search(self):
        query = self.input_entry.get().strip()
        if query:
            similar = self.services.knowledge.get_similar(query)
            self.display_response("\n".join(f"[{s:.2f}] {q}: {r}" for q, r, s in similar) or "Нет данных")

    def show_skills(self):
        self.display_response(f"Умения:\n- Генерация текста с учетом эмоций\n- Обработка кода\n- Работа с интернет-ресурсами\n- Обнаружение ошибок\n- Редактирование кода\n- Дублирование структуры\n- Инспекция кода\n- Обучение моделей\n- Визуализация данных\n\nТекущий уровень опыта: {self.services.knowledge.learning_rate:.1f}%")

    def _get_context(self) -> str:
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.context)

    def run(self):
        if self.initialized:
            logging.info("Запуск основного цикла приложения")
            self.mainloop()
        else:
            logging.error("Приложение не инициализировано, запуск невозможен")

class APISettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.title("Настройка API")
        self.geometry("300x250")
        self.config = config
        self._init_ui()

    def _init_ui(self):
        ctk.CTkLabel(self, text="YandexGPT key:").grid(row=0, column=0, padx=5, pady=5)
        self.key_entry = ctk.CTkEntry(self, width=150)
        self.key_entry.grid(row=0, column=1, padx=5, pady=5)
        self.key_entry.insert(0, self.config.get_key())

        ctk.CTkButton(self, text="📋 Вставить", command=self._paste_key,
                     width=80, fg_color="#1C2526", hover_color="#4A4A4A").grid(row=0, column=2, padx=5)

        ctk.CTkLabel(self, text="Folder ID:").grid(row=1, column=0, padx=5, pady=5)
        self.folder_entry = ctk.CTkEntry(self, width=150)
        self.folder_entry.grid(row=1, column=1, padx=5, pady=5)
        self.folder_entry.insert(0, self.config.get_folder_id())

        ctk.CTkButton(self, text="Сохранить", command=self._save_api_key,
                     fg_color="#1C2526", hover_color="#4A4A4A").grid(row=2, column=0, columnspan=3, pady=10)

        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)

    def _paste_key(self):
        try:
            clipboard_text = self.clipboard_get()
            if clipboard_text:
                self.key_entry.delete(0, "end")
                self.key_entry.insert(0, clipboard_text)
                self.status_label.configure(text="Текст вставлен из буфера")
            else:
                self.status_label.configure(text="Буфер обмена пуст")
        except tk.TclError:
            self.status_label.configure(text="Ошибка доступа к буферу")

    def _save_api_key(self):
        key = self.key_entry.get().strip()
        folder_id = self.folder_entry.get().strip()

        if not key or not folder_id:
            self.status_label.configure(text="Ошибка: Поля не могут быть пустыми")
            return

        if not validate_folder_id(folder_id):
            self.status_label.configure(text="Ошибка: folder_id должен быть 20 символов (буквы/цифры)")
            return

        temp_gpt = YandexGPT(key, folder_id)
        is_valid, status_message = temp_gpt.check_availability()
        self.status_label.configure(text=status_message)

        if is_valid:
            self.config.update_api_key("gpt_key_1", key)
            self.config.update_folder_id(folder_id)
            self.status_label.configure(text="Настройки сохранены")
            self.after(1000, self.destroy)

class APIKeyCheckWindow(ctk.CTkToplevel):
    def __init__(self, parent, services, config):
        super().__init__(parent)
        self.title("Проверка API ключа")
        self.geometry("400x300")
        self.services = services
        self.config = config
        self._init_ui()

    def _init_ui(self):
        ctk.CTkLabel(self, text="Введите ключ:").grid(row=0, column=0, padx=5, pady=5)
        self.key_entry = ctk.CTkEntry(self, width=200)
        self.key_entry.grid(row=0, column=1)
        ctk.CTkLabel(self, text="Folder ID:").grid(row=1, column=0, padx=5, pady=5)
        self.folder_entry = ctk.CTkEntry(self, width=200)
        self.folder_entry.grid(row=1, column=1)
        self.folder_entry.insert(0, self.config.get_folder_id())
        ctk.CTkButton(self, text="Проверить", command=self._check_key).grid(row=2, column=0, columnspan=2, pady=10)
        self.status_text = ctk.CTkTextbox(self, width=350, height=100)
        self.status_text.grid(row=3, column=0, columnspan=2)

    def _check_key(self):
        key = self.key_entry.get()
        folder_id = self.folder_entry.get()
        if key and validate_folder_id(folder_id):
            gpt = YandexGPT(key, folder_id)
            available, status = gpt.check_availability()
            self.status_text.delete("1.0", "end")
            self.status_text.insert("1.0", f"Статус: {status}")

if __name__ == "__main__":
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        logging.error(f"Ошибка загрузки nltk данных: {e}")
    app = NereMoreInterface()
    app.run()
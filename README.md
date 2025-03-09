import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
import random
import base64
from gtts import gTTS
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

# Импорты Flax и JAX
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random
import optax
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("nere_more.log"), logging.StreamHandler()],
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Модель на Flax для анализа настроений
class SentimentFlaxModel(nn.Module):
    hidden_dim: int = 128
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x)  # Ограничиваем выход в диапазон [-1, 1]
        return x

# Класс для анализа настроений с использованием Flax/JAX
class SentimentAnalyzer:
    def __init__(self, vectorizer=None, model_path="sentiment_model.pkl"):
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        self.model_path = model_path
        self.rng = random.PRNGKey(42)
        self.model = SentimentFlaxModel()
        self.params = None
        self.opt_state = None
        self.optimizer = optax.adam(learning_rate=0.001)
        self.mood_history = deque(maxlen=50)
        self._init_model()
        self._load_or_train_model()

    def _init_model(self):
        dummy_input = jnp.zeros((1, 5000))
        self.params = self.model.init(self.rng, dummy_input, training=False)['params']
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.optimizer
        )

    def _loss_fn(self, params, batch_x, batch_y):
        preds = self.model.apply({'params': params}, batch_x, training=True, rngs={'dropout': random.PRNGKey(0)})
        loss = jnp.mean((preds - batch_y) ** 2)  # MSE Loss
        return loss

    def _train_step(self, state, batch_x, batch_y):
        loss, grads = jax.value_and_grad(self._loss_fn)(state.params, batch_x, batch_y)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def _load_or_train_model(self):
        texts = [
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
        ]
        scores = [1.0, -1.0, 0.0, -0.6, -0.9, 0.8, -0.2, -0.8, 1.0, -0.4]
        X = self.vectorizer.fit_transform(texts).toarray()
        y = jnp.array(scores).reshape(-1, 1)
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_size = int(0.8 * n_samples)
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = jnp.array(X_train), jnp.array(X_test)
        y_train, y_test = jnp.array(y_train), jnp.array(y_test)
        epochs = 100
        batch_size = 4
        num_batches = len(X_train) // batch_size
        for epoch in range(epochs):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            epoch_loss = 0.0
            for batch in range(num_batches):
                batch_idx = indices[batch * batch_size:(batch + 1) * batch_size]
                batch_x = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                self.state, loss = self._train_step(self.state, batch_x, batch_y)
                epoch_loss += loss
            if epoch % 20 == 0:
                logging.info(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.4f}")
        joblib.dump(self.state.params, self.model_path)

    def predict_sentiment_score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        try:
            x = self.vectorizer.transform([text]).toarray()
            x = jnp.array(x)
            pred = self.model.apply({'params': self.state.params}, x, training=False)
            score = float(pred[0, 0])
            self.mood_history.append(score)
            return score
        except Exception as e:
            logging.error(f"Ошибка предсказания настроения: {e}")
            return 0.0

    def analyze_mood_trend(self) -> Dict[str, Any]:
        if not self.mood_history:
            return {"current_mood": 0.0, "average_mood": 0.0, "trend": "нет данных"}
        current_mood = self.mood_history[-1] if self.mood_history else 0.0
        average_mood = sum(self.mood_history) / len(self.mood_history)
        if len(self.mood_history) >= 2:
            recent_moods = list(self.mood_history)[-5:]
            if all(recent_moods[i] <= recent_moods[i+1] for i in range(len(recent_moods)-1)):
                trend = "рост настроения"
            elif all(recent_moods[i] >= recent_moods[i+1] for i in range(len(recent_moods)-1)):
                trend = "спад настроения"
            else:
                trend = "стабильное настроение"
        else:
            trend = "недостаточно данных для анализа тенденции"
        return {"current_mood": current_mood, "average_mood": average_mood, "trend": trend}

    def interpret_mood(self, mood_score: float) -> str:
        if mood_score > 0.5:
            return "позитивное"
        elif mood_score < -0.5:
            return "негативное"
        else:
            return "нейтральное"

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
        self.sentiment_analyzer = SentimentAnalyzer()

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

    def analyze_comments_with_flax(self, code: str) -> Dict[str, float]:
        comments = []
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                comment = line[1:].strip()
                if comment:
                    comments.append(comment)
            elif '#' in line:
                comment = line[line.find('#')+1:].strip()
                if comment:
                    comments.append(comment)
        sentiment_scores = {}
        for comment in comments:
            score = self.sentiment_analyzer.predict_sentiment_score(comment)
            sentiment_scores[comment] = score
            logging.info(f"Flax/JAX анализ комментария: {comment} -> оценка: {score:.2f}")
        return sentiment_scores

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
            headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {self.api_key}"}
            payload = {
                "modelUri": f"gpt://{self.folder_id}/{self.model}",
                "completionOptions": {"maxTokens": self.max_tokens, "temperature": self.temperature},
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
            headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {self.api_key}"}
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
        self.params = None
        self.opt_state = None
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.key = random.PRNGKey(42)
        logging.info("Инициализация DataModelTrainer с JAX")

    def generate_synthetic_data(self, n_samples=1000):
        X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = jnp.array(self.X_train)
        self.X_test = jnp.array(self.X_test)
        self.y_train = jnp.array(self.y_train)
        self.y_test = jnp.array(self.y_test)
        logging.info("Сгенерированы синтетические данные для JAX")

    def augment_data(self, X, noise_factor=0.1):
        key = random.PRNGKey(int(time.time()))
        noise = random.normal(key, X.shape) * noise_factor
        return X + noise

    def init_model(self, input_dim, hidden_dims=[64, 32], l2_lambda=0.01):
        params = []
        key = self.key
        for i in range(len(hidden_dims) + 1):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            out_dim = hidden_dims[i] if i < len(hidden_dims) else 1
            key, subkey = random.split(key)
            W = random.normal(subkey, (in_dim, out_dim)) * jnp.sqrt(2/in_dim)
            b = jnp.zeros(out_dim)
            scale = jnp.ones(out_dim)
            shift = jnp.zeros(out_dim)
            params.append({'W': W, 'b': b, 'scale': scale, 'shift': shift})
        self.params = params
        self.l2_lambda = l2_lambda

    def batch_norm(self, x, scale, shift, eps=1e-5):
        mean = jnp.mean(x, axis=0)
        var = jnp.var(x, axis=0)
        x_norm = (x - mean) / jnp.sqrt(var + eps)
        return x_norm * scale + shift

    @partial(jit, static_argnums=(0,))
    def forward(self, params, x):
        for i, layer in enumerate(params[:-1]):
            x = jnp.dot(x, layer['W']) + layer['b']
            x = self.batch_norm(x, layer['scale'], layer['shift'])
            x = jax.nn.relu(x)
        x = jnp.dot(x, params[-1]['W']) + params[-1]['b']
        return jax.nn.sigmoid(x)

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, X, y):
        preds = self.forward(params, X)
        bce_loss = -jnp.mean(y * jnp.log(preds + 1e-7) + (1 - y) * jnp.log(1 - preds + 1e-7))
        l2_loss = self.l2_lambda * sum(jnp.sum(jnp.square(p['W'])) for p in params)
        return bce_loss + l2_loss

    def accuracy(self, params, X, y):
        preds = self.forward(params, X)
        return jnp.mean((preds > 0.5) == y)

    def build_model(self, optimizer_name="Adam", learning_rate=0.001):
        if optimizer_name == "Adam":
            optimizer = optax.adam(learning_rate=learning_rate)
        else:
            optimizer = optax.sgd(learning_rate=learning_rate)
        self.opt_state = optimizer.init(self.params)
        self.optimizer = optimizer
        logging.info(f"Модель создана с JAX и оптимизатором {optimizer_name}")

    def train_model(self, epochs=50, batch_size=32):
        if self.params is None or self.X_train is None:
            raise ValueError("Модель или данные не инициализированы")
        grad_loss = grad(self.loss_fn)
        num_batches = len(self.X_train) // batch_size
        for epoch in range(epochs):
            X_train_aug = self.augment_data(self.X_train)
            indices = jnp.arange(len(self.X_train))
            self.key, subkey = random.split(self.key)
            indices = random.permutation(subkey, indices)
            epoch_loss = 0.0
            for batch in range(num_batches):
                batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
                X_batch = X_train_aug[batch_indices]
                y_batch = self.y_train[batch_indices]
                grads = grad_loss(self.params, X_batch, y_batch)
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
                self.params = optax.apply_updates(self.params, updates)
                batch_loss = self.loss_fn(self.params, X_batch, y_batch)
                epoch_loss += batch_loss
            train_loss = epoch_loss / num_batches
            val_loss = self.loss_fn(self.params, self.X_test, self.y_test)
            train_acc = self.accuracy(self.params, self.X_train, self.y_train)
            val_acc = self.accuracy(self.params, self.X_test, self.y_test)
            self.history['loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['accuracy'].append(float(train_acc))
            self.history['val_accuracy'].append(float(val_acc))
            logging.info(f"Эпоха {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    def hyperparameter_tuning(self, learning_rates=[1e-3, 1e-2], batch_sizes=[16, 32], epochs=50):
        best_params = None
        best_val_acc = 0.0
        results = []
        for lr in learning_rates:
            for bs in batch_sizes:
                logging.info(f"Тестирование lr={lr}, batch_size={bs}")
                self.init_model(input_dim=self.X_train.shape[1])
                self.build_model(learning_rate=lr)
                self.train_model(epochs=epochs, batch_size=bs)
                val_acc = self.history['val_accuracy'][-1]
                results.append({'lr': lr, 'batch_size': bs, 'val_acc': val_acc})
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {'lr': lr, 'batch_size': bs}
        logging.info(f"Лучшие параметры: lr={best_params['lr']}, batch_size={best_params['batch_size']}, val_acc={best_val_acc:.4f}")
        return best_params

    def visualize_results(self, output_path="training_results.png"):
        if not self.history['loss']:
            logging.error("История обучения отсутствует")
            return
        history_df = pd.DataFrame(self.history)
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
        self.sentiment_analyzer = SentimentAnalyzer()

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
            score = self.sentiment_analyzer.predict_sentiment_score(last_input)
            self.user_mood = score
            logging.info(f"Flax/JAX анализ: {last_input} -> оценка: {score:.2f}")
        if time.time() - self.last_interaction > 60:
            if not self.questions:
                self.gui.display_response("Нет доступных вопросов.")
                return
            question = random.choice(self.questions)
            self.gui.display_response(question)
        else:
            if not self.greetings:
                self.gui.display_response("Привет! Чем могу помочь?")
                return
            if self.user_mood > 0.5:
                greeting = random.choice(self.greetings) + " Ты выглядишь счастливым!"
            elif self.user_mood < -0.5:
                greeting = random.choice(self.greetings) + " Не грусти, давай что-нибудь сделаем!"
            else:
                greeting = random.choice(self.greetings)
            if not self.suggestions:
                suggestion = "Давай что-нибудь сделаем!"
            else:
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
        self.sentiment_analyzer = SentimentAnalyzer()
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
            self.data_trainer.init_model(input_dim=self.data_trainer.X_train.shape[1])
            best_params = self.data_trainer.hyperparameter_tuning()
            self.data_trainer.build_model(optimizer_name=optimizer_name, learning_rate=best_params['lr'])
            self.data_trainer.train_model(epochs=epochs, batch_size=best_params['batch_size'])
            self.data_trainer.visualize_results()
            return (f"Модель обучена с JAX! Лучшие параметры: lr={best_params['lr']}, batch_size={best_params['batch_size']}\n"
                    f"Графики сохранены в training_results.png\nОпыт ИИ: {self.knowledge.learning_rate:.1f}%")
        except Exception as e:
            logging.error(f"Ошибка при обучении модели: {e}")
            return f"Ошибка при обучении: {str(e)}"

    def suggest_action_algorithm(self, query: str, user_emotion: Optional[float] = None) -> str:
        if user_emotion is None:
            user_emotion = self.sentiment_analyzer.predict_sentiment_score(query)
            logging.info(f"Flax/JAX анализ запроса: {query} -> оценка: {user_emotion:.2f}")
        mood_analysis = self.sentiment_analyzer.analyze_mood_trend()
        mood_interpretation = self.sentiment_analyzer.interpret_mood(user_emotion)
        avg_mood_interpretation = self.sentiment_analyzer.interpret_mood(mood_analysis["average_mood"])
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
        if mood_analysis["trend"] == "спад настроения":
            suggestion += " Замечаю, что настроение немного ухудшилось, давайте попробуем что-то позитивное!"
        elif mood_analysis["trend"] == "рост настроения":
            suggestion += " Отлично, настроение улучшается, давайте продолжим!"
        built_context = self.knowledge.build_context(query)
        prompt = {
            "modelUri": f"gpt://{self.config.get_folder_id()}/{self.gpt.model}",
            "completionOptions": {"stream": False, "temperature": 0.5, "maxTokens": 2000},
            "messages": [
                {"role": "system", "text": f"Ты ассистент с уровнем опыта {self.knowledge.learning_rate:.1f}%. Используй накопленные знания и адаптируйся к эмоциональному состоянию пользователя (тон: {tone})."},
                {"role": "user", "text": f"Контекст:\n{memory_context}\n\nЗапрос: {query}\n\nОсновной фокус: {main_focus}"}
            ]
        }
        response = self.gpt.invoke(prompt)
        resp_score = self.sentiment_analyzer.predict_sentiment_score(response)
        logging.info(f"Flax/JAX анализ ответа: {response[:50]}... -> оценка: {resp_score:.2f}")
        self.knowledge.save(query, response, context=f"Эмоциональный тон: {tone}, Фокус: {main_focus}")
        mood_summary = (
            f"Анализ настроения:\n"
            f"- Текущее настроение по запросу: {mood_interpretation} (оценка: {user_emotion:.2f})\n"
            f"- Среднее настроение за сессию: {avg_mood_interpretation} (оценка: {mood_analysis['average_mood']:.2f})\n"
            f"- Тенденция настроения: {mood_analysis['trend']}\n"
        )
        return (
            f"Основной фокус: {main_focus}\n"
            f"Эмоциональный тон ответа: {tone}\n"
            f"Предложение: {suggestion}\n"
            f"Ответ: {response}\n"
            f"Оценка настроения ответа: {resp_score:.2f}\n"
            f"{mood_summary}"
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
                sentiment_scores = self.code_optimizer.analyze_comments_with_flax(query)
                comment_analysis = "\n".join(
                    [f"- Комментарий '{comment}' -> настроение: {self.sentiment_analyzer.interpret_mood(score)} (оценка: {score:.2f})"
                     for comment, score in sentiment_scores.items()]
                ) if sentiment_scores else "Комментариев для анализа нет."
                response = (f"Отформатированный код:\n{formatted_code}\n\n"
                            f"Классификация:\n- Назначение: {purpose}\n- Место: {location}\n\n"
                            f"Ошибки:\n{chr(10).join(errors)}\n\n"
                            f"Анализ настроения в комментариях:\n{comment_analysis}")
                self.knowledge.save(query, response)
                user_emotion = self.sentiment_analyzer.predict_sentiment_score(query)
                mood_analysis = self.sentiment_analyzer.analyze_mood_trend()
                mood_interpretation = self.sentiment_analyzer.interpret_mood(user_emotion)
                avg_mood_interpretation = self.sentiment_analyzer.interpret_mood(mood_analysis["average_mood"])
                mood_summary = (
                    f"Анализ настроения:\n"
                    f"- Текущее настроение по запросу: {mood_interpretation} (оценка: {user_emotion:.2f})\n"
                    f"- Среднее настроение за сессию: {avg_mood_interpretation} (оценка: {mood_analysis['average_mood']:.2f})\n"
                    f"- Тенденция настроения: {mood_analysis['trend']}\n"
                )
                return f"{response}\n\n{mood_summary}[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]"
            except Exception as e:
                return f"Ошибка обработки кода: {e}"
        built_context = self.knowledge.build_context(query)
        prompt = {
            "modelUri": f"gpt://{self.config.get_folder_id()}/{self.gpt.model}",
            "completionOptions": {"stream": False, "temperature": max(0.3, 0.6 - (self.knowledge.learning_rate / 200)), "maxTokens": 2000},
            "messages": [
                {"role": "system", "text": f"Ты ассистент, который учится на основе опыта (текущий уровень: {self.knowledge.learning_rate:.1f}%). Используй накопленные знания для логически последовательных ответов."},
                {"role": "user", "text": f"Контекст:\n{built_context}\n\nТекущий запрос: {query}"}
            ]
        }
        response = self.gpt.invoke(prompt)
        resp_score = self.sentiment_analyzer.predict_sentiment_score(response)
        logging.info(f"Flax/JAX анализ ответа: {response[:50]}... -> оценка: {resp_score:.2f}")
        self.knowledge.save(query, response, context=built_context)
        user_emotion = self.sentiment_analyzer.predict_sentiment_score(query)
        mood_analysis = self.sentiment_analyzer.analyze_mood_trend()
        mood_interpretation = self.sentiment_analyzer.interpret_mood(user_emotion)
        avg_mood_interpretation = self.sentiment_analyzer.interpret_mood(mood_analysis["average_mood"])
        mood_summary = (
            f"Анализ настроения:\n"
            f"- Текущее настроение по запросу: {mood_interpretation} (оценка: {user_emotion:.2f})\n"
            f"- Среднее настроение за сессию: {avg_mood_interpretation} (оценка: {mood_analysis['average_mood']:.2f})\n"
            f"- Тенденция настроения: {mood_analysis['trend']}\n"
        )
        return f"{response}\n\nОценка настроения ответа: {resp_score:.2f}\n{mood_summary}[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]"

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
            errors = self.services.code_optimizer.detect_errors(code)
            sentiment_scores = self.services.code_optimizer.analyze_comments_with_flax(code)
            comment_analysis = "\n".join(
                [f"- Комментарий '{comment}' -> настроение: {self.services.sentiment_analyzer.interpret_mood(score)} (оценка: {score:.2f})"
                 for comment, score in sentiment_scores.items()]
            ) if sentiment_scores else "Комментариев для анализа нет."
            self.code_textbox.delete("1.0", "end")
            self.code_textbox.insert("1.0", f"Код:\n{code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}\n\nОшибки:\n{chr(10).join(errors)}\n\nАнализ настроения в комментариях:\n{comment_analysis}")

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
            sentiment_scores = self.services.code_optimizer.analyze_comments_with_flax(content)
            comment_analysis = "\n".join(
                [f"- Комментарий '{comment}' -> настроение: {self.services.sentiment_analyzer.interpret_mood(score)} (оценка: {score:.2f})"
                 for comment, score in sentiment_scores.items()]
            ) if sentiment_scores else "Комментариев для анализа нет."
            response = (f"Код вставлен:\n{formatted_code}\n\n"
                        f"Классификация:\n- Назначение: {purpose}\n- Место: {location}\n\n"
                        f"Ошибки:\n{chr(10).join(errors)}\n\n"
                        f"Анализ настроения в комментариях:\n{comment_analysis}")
            self.services.knowledge.save(f"Inserted code (ID: {uuid.uuid4().hex[:8]})", formatted_code)
            user_emotion = self.services.sentiment_analyzer.predict_sentiment_score(content)
            mood_analysis = self.services.sentiment_analyzer.analyze_mood_trend()
            mood_interpretation = self.services.sentiment_analyzer.interpret_mood(user_emotion)
            avg_mood_interpretation = self.services.sentiment_analyzer.interpret_mood(mood_analysis["average_mood"])
            mood_summary = (
                f"Анализ настроения:\n"
                f"- Текущее настроение по тексту: {mood_interpretation} (оценка: {user_emotion:.2f})\n"
                f"- Среднее настроение за сессию: {avg_mood_interpretation} (оценка: {mood_analysis['average_mood']:.2f})\n"
                f"- Тенденция настроения: {mood_analysis['trend']}\n"
            )
            self.display_response(f"{response}\n\n{mood_summary}")
        else:
            self.services.knowledge.save(f"Inserted text (ID: {uuid.uuid4().hex[:8]})", content)
            self.display_response(f"Вставлено: {content[:100]}...")

    def _magnet_search(self):
        query = self.input_entry.get().strip()
        if query:
            similar = self.services.knowledge.get_similar(query)
            self.display_response("\n".join(f"[{s:.2f}] {q}: {r}" for q, r, s in similar) or "Нет данных")

    def show_skills(self):
        self.display_response(f"Умения:\n- Генерация текста с учетом эмоций\n- Обработка кода\n- Работа с интернет-ресурсами\n- Обнаружение ошибок\n- Редактирование кода\n- Дублирование структуры\n- Инспекция кода\n- Обучение моделей с JAX\n- Визуализация данных\n\nТекущий уровень опыта: {self.services.knowledge.learning_rate:.1f}%")

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
    app = NereMoreInterface()
    app.run()
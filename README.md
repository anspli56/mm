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
import os
from gtts import gTTS
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("nere_more.log"), logging.StreamHandler()],
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def validate_folder_id(folder_id: str) -> bool:
    """Проверяет валидность folder_id."""
    return bool(re.match(r'^[a-zA-Z0-9]{20}$', folder_id))

def fibonacci(n: int) -> int:
    """Вычисляет n-е число Фибоначчи для формирования последовательности опыта."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class AudioManager:
    def play_sound(self, text: str, filename: str) -> None:
        """Воспроизводит текст как аудио на русском языке."""
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
        """Загружает конфигурацию из файла или создает новую."""
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
                    self.config = self.default_config | self.config
            else:
                self.config = self.default_config
                self._save_config()
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            self.config = self.default_config
            self._save_config()

    def _save_config(self):
        """Сохраняет конфигурацию в файл."""
        try:
            with open("nere_more_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except IOError as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")

    def _validate_and_update_on_startup(self):
        """Проверяет и обновляет конфигурацию при запуске."""
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
        """Возвращает расшифрованный API ключ."""
        for key in self.config["yandex"]["keys"]:
            if key["type"] == "gpt":
                try:
                    return self._cipher.decrypt(key["value"].encode()).decode()
                except Exception:
                    return key["value"]
        return ""

    def update_api_key(self, key_id: str, value: str) -> bool:
        """Обновляет API ключ после проверки."""
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
        """Возвращает folder_id."""
        return self.config["yandex"].get("folder_id", "")

    def update_folder_id(self, folder_id: str) -> bool:
        """Обновляет folder_id после проверки."""
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
        """Классифицирует код по назначению и местоположению."""
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
        """Обнаруживает ошибки в коде."""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Синтаксическая ошибка: {str(e)}")
        return errors if errors else ["Ошибок не обнаружено"]

    def analyze_structure(self, code: str) -> Dict[str, List[str]]:
        """Анализирует структуру кода."""
        tree = ast.parse(code)
        structure = {"functions": [], "classes": []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append(node.name)
        
        return structure

    def suggest_structure(self, code: str, errors: List[str]) -> str:
        """Предлагает улучшения структуры кода."""
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
        """Дублирует структуру кода."""
        tree = ast.parse(code)
        duplicated_code = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                duplicated_code.append(ast.unparse(node))
        
        return "\n\n".join(duplicated_code) if duplicated_code else code

    def suggest_integration_points(self, code: str, location: str) -> List[Tuple[str, str]]:
        """Предлагает точки интеграции для кода."""
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
        """Инициализирует YandexGPT с API ключом и folder_id."""
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
        """Проверяет валидность учетных данных."""
        if not self.api_key or len(self.api_key.strip()) < 10:
            self.status = "Ошибка: API-ключ пустой или слишком короткий"
            return
        if not validate_folder_id(self.folder_id):
            self.status = "Ошибка: folder_id должен быть 20 символов (буквы/цифры)"
            return
        self.check_availability()

    def check_availability(self) -> Tuple[bool, str]:
        """Проверяет доступность API."""
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
        """Выполняет запрос к YandexGPT."""
        if not self.available:
            return f"API отключен: {self.status}"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}"
            }
            
            response = requests.post(self.url, headers=headers, json=json_payload)
            if response.status_code != 200:
                return f"Ошибка: {response.status_code} {response.reason}"
            
            result = response.json()
            return result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "No data")
                
        except Exception as e:
            logging.error(f"Ошибка Yandex GPT: {e}")
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
        self.context_history = deque(maxlen=50)  # Хранит историю для логической последовательности
        self._load_state()

    def _load_state(self):
        """Загружает состояние векторайзера и кластеризации."""
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
                    embeddings = [np.frombuffer(entry['embeddings'], dtype=np.float16) for entry in self.db.all() if 'embeddings' in entry]
                    if embeddings:
                        self.kmeans.fit(np.stack(embeddings))
                    self._save_state()
            self.experience_level = len(self.db.all())
            self.learning_rate = self.config["learning"]["learning_rate"]
        except Exception as e:
            logging.error(f"Ошибка загрузки состояния: {e}")
            self.vectorizer_fitted = False

    def _save_state(self):
        """Сохраняет состояние моделей."""
        if self.vectorizer_fitted:
            try:
                joblib.dump(self.vectorizer, "vectorizer.pkl")
                joblib.dump(self.kmeans, "kmeans.pkl")
            except Exception as e:
                logging.error(f"Ошибка сохранения моделей: {e}")

    def _ensure_vectorizer_fitted(self, text: str):
        """Обеспечивает обучение векторайзера."""
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
        """Обновляет уровень опыта и процент обучения на основе числа записей."""
        self.experience_level += 1
        fib_value = fibonacci(min(self.experience_level, 20))  # Ограничиваем рост для реалистичности
        self.learning_rate = min(100.0, fib_value * 0.5)  # Увеличиваем медленнее, до 100%
        self.config["learning"]["learning_rate"] = self.learning_rate
        Config()._save_config()
        return self.learning_rate

    def save(self, query: str, response: str, context: str = "", feedback: float = 0.0):
        """Сохраняет запрос и ответ в базу знаний, обновляя опыт."""
        with self._lock:
            if len(response.strip()) < 5:
                return
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.query_cache[query_hash] = response
            self._ensure_vectorizer_fitted(response)
            query_vec = self.vectorizer.transform([query]).toarray().astype(np.float16)[0]
            embeddings = self.vectorizer.transform([response]).toarray().astype(np.float16)[0]
            entry = {
                'id': str(uuid.uuid4()),
                'query': query,
                'response': response,
                'context': context,
                'embeddings': embeddings.tobytes(),
                'feedback': feedback,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'learning_rate': self.update_experience()
            }
            self.db.insert(entry)
            self.context_history.append(f"Вопрос: {query}\nОтвет: {response}")

    def get_similar(self, query: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Возвращает похожие записи из базы знаний."""
        self._ensure_vectorizer_fitted(query)
        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float16)[0]
        all_data = [(entry['query'], entry['response'], np.frombuffer(entry['embeddings'], dtype=np.float16))
                    for entry in self.db.all() if 'embeddings' in entry]
        if not all_data:
            return []
        embeddings = np.stack([item[2] for item in all_data])
        similarities = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
        results = sorted(zip(all_data, similarities), key=lambda x: x[1], reverse=True)[:top_n]
        return [(q, r, float(s)) for (q, r, _), s in results]

    def save_web_content(self, url: str, query: str) -> bool:
        """Сохраняет контент с веб-страницы и обновляет опыт."""
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
        """Формирует контекст на основе накопленного опыта."""
        similar = self.get_similar(query, top_n=3)
        context = "\n\n".join([f"Ранее: {q}\nОтвет: {r} (сходство: {s:.2f})" for q, r, s in similar])
        if not context:
            context = "У меня пока мало опыта по этому вопросу, но я постараюсь ответить на основе доступных данных."
        history = "\n".join(list(self.context_history)[-5:])  # Последние 5 записей для последовательности
        return f"История взаимодействия:\n{history}\n\nПрошлый опыт:\n{context}"

    def _clear_cache(self, text_id: str = None):
        """Очищает кэш базы знаний."""
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

class YandexAIServices:
    def __init__(self, gui_parent=None):
        self.config = Config()
        self.gui_parent = gui_parent
        self._request_credentials_if_needed()
        self.knowledge = KnowledgeBase(self)
        self.gpt = YandexGPT(self.config.get_key(), self.config.get_folder_id())
        self.code_optimizer = CodeOptimizationModule(self.config)
        
        available, status = self.gpt.check_availability()
        if not available:
            logging.warning(f"Инициализация с проблемой API: {status}")
            if gui_parent:
                gui_parent.status_label.configure(text=f"Ошибка API: {status}")

    def _request_credentials_if_needed(self):
        """Запрашивает учетные данные, если они отсутствуют."""
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

    def generate_response(self, query: str, context: str = "") -> str:
        """Генерирует ответ на основе опыта и интернет-ресурсов."""
        if not query:
            return "Ошибка: Запрос пуст"
        
        # Обработка URL
        urls = re.findall(r'https?://\S+', query)
        if urls:
            success = self.knowledge.save_web_content(urls[0], query)
            return f"Сохранено с {urls[0]}\n[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]" if success else f"Ошибка с {urls[0]}"
        
        # Обработка кода
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
        
        # Формирование ответа на основе опыта
        built_context = self.knowledge.build_context(query)
        prompt = {
            "modelUri": f"gpt://{self.config.get_folder_id()}/{self.gpt.model}",
            "completionOptions": {
                "stream": False,
                "temperature": max(0.3, 0.6 - (self.knowledge.learning_rate / 200)),  # Температура снижается с опытом
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
        self.knowledge.save(query, response, context=built_context)
        return f"{response}\n\n[Опыт ИИ: {self.knowledge.learning_rate:.1f}%]"

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
        self.geometry("1000x700")
        self.services = services
        self.parent = parent
        self.original_code = ""
        self._init_ui()
        self._configure_syntax_highlighting()

    def _init_ui(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="#1C2526")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.code_frame = ctk.CTkFrame(self.main_frame, fg_color="#1C2526")
        self.code_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.code_entry = tk.Text(self.code_frame, height=25, width=40, bg="#1C2526", fg="#FFFFFF", insertbackground="white", font=("Courier", 12))
        self.code_entry.pack(fill="both", expand=True, pady=5)
        self.code_entry.insert("1.0", "# Введите код здесь\n")
        self.code_entry.bind("<KeyRelease>", self._update_output)
        self.original_code = self.code_entry.get("1.0", "end-1c").strip()

        self.output_frame = ctk.CTkFrame(self.main_frame, fg_color="#1C2526")
        self.output_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.formatted_label = ctk.CTkLabel(self.output_frame, text="Отформатированный код:", text_color="#FFFFFF")
        self.formatted_label.pack(pady=2)
        self.formatted_output = tk.Text(self.output_frame, height=15, width=40, bg="#1C2526", fg="#FFFFFF", font=("Courier", 12))
        self.formatted_output.pack(fill="both", expand=True, pady=5)

        self.changes_label = ctk.CTkLabel(self.output_frame, text="Изменения и интеграция:", text_color="#FFFFFF")
        self.changes_label.pack(pady=2)
        self.changes_output = tk.Text(self.output_frame, height=15, width=40, bg="#1C2526", fg="#FFFFFF", font=("Courier", 12))
        self.changes_output.pack(fill="both", expand=True, pady=5)

        button_frame = ctk.CTkFrame(self.main_frame, fg_color="#2F3536")
        button_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(button_frame, text="Сохранить", command=self._save_code, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Проверить", command=self._inspect_code, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Применить", command=self._apply_to_app, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Дублировать", command=self._duplicate_structure, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Отмена", command=self.destroy, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)

    def _configure_syntax_highlighting(self):
        self.code_entry.tag_configure("keyword", foreground="#FF5555")
        self.code_entry.tag_configure("string", foreground="#55FF55")
        self.code_entry.tag_configure("comment", foreground="#888888")
        self.formatted_output.tag_configure("keyword", foreground="#FF5555")
        self.formatted_output.tag_configure("string", foreground="#55FF55")
        self.formatted_output.tag_configure("comment", foreground="#888888")
        self.changes_output.tag_configure("keyword", foreground="#FF5555")
        self.changes_output.tag_configure("string", foreground="#55FF55")
        self.changes_output.tag_configure("comment", foreground="#888888")
        self.changes_output.tag_configure("integration", background="#4444FF", foreground="#FFFFFF")
        self.changes_output.tag_configure("change", background="#FF4444", foreground="#FFFFFF")

    def _highlight_syntax(self, text_widget, code):
        text_widget.mark_set("range_start", "1.0")
        
        for tag in ("keyword", "string", "comment", "integration", "change"):
            text_widget.tag_remove(tag, "1.0", "end")

        keywords = {"def", "class", "if", "else", "for", "while", "import", "from", "return", "try", "except"}
        for word in keywords:
            start = "1.0"
            while True:
                pos = text_widget.search(r"\m" + word + r"\M", start, stopindex="end", regexp=True)
                if not pos:
                    break
                text_widget.tag_add("keyword", pos, f"{pos}+{len(word)}c")
                start = f"{pos}+{len(word)}c"

        for match in re.finditer(r'["\'].*?["\']', code):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            text_widget.tag_add("string", start, end)

        for match in re.finditer(r"#.*$", code, re.MULTILINE):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            text_widget.tag_add("comment", start, end)

    def _update_output(self, event=None):
        self._highlight_syntax(self.code_entry, self.code_entry.get("1.0", "end-1c"))
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            self.formatted_output.delete("1.0", "end")
            self.formatted_output.insert("1.0", "Код пуст")
            self.changes_output.delete("1.0", "end")
            self.changes_output.insert("1.0", "Нет кода для анализа")
            return

        try:
            formatted_code = black.format_str(code, mode=black.FileMode())
        except Exception as e:
            formatted_code = code + f"\n# Ошибка форматирования: {e}"
        
        purpose, location = self.services.code_optimizer.classify_code(code)
        errors = self.services.code_optimizer.detect_errors(code)
        integration_points = self.services.code_optimizer.suggest_integration_points(code, location)

        self.formatted_output.delete("1.0", "end")
        self.formatted_output.insert("1.0", formatted_code)
        self._highlight_syntax(self.formatted_output, formatted_code)

        changes_text = formatted_code
        original_lines = self.original_code.splitlines()
        new_lines = formatted_code.splitlines()
        
        for i, (orig, new) in enumerate(zip(original_lines, new_lines)):
            if orig != new:
                start_pos = f"{i+1}.0"
                end_pos = f"{i+1}.end"
                self.changes_output.tag_add("change", start_pos, end_pos)

        for name, suggestion in integration_points:
            for i, line in enumerate(new_lines):
                if name in line:
                    start_pos = f"{i+1}.0"
                    end_pos = f"{i+1}.end"
                    self.changes_output.tag_add("integration", start_pos, end_pos)
                    changes_text += f"\n# {suggestion}"

        self.changes_output.delete("1.0", "end")
        self.changes_output.insert("1.0", changes_text)
        self._highlight_syntax(self.changes_output, changes_text)

    def _inspect_code(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            messagebox.showwarning("Предупреждение", "Код пустой")
            return

        purpose, location = self.services.code_optimizer.classify_code(code)
        errors = self.services.code_optimizer.detect_errors(code)
        structure = self.services.code_optimizer.analyze_structure(code)
        suggestions = self.services.code_optimizer.suggest_structure(code, errors)
        integration_points = self.services.code_optimizer.suggest_integration_points(code, location)

        inspection_text = (
            f"Инспекция кода:\n"
            f"Классификация: {purpose} ({location})\n"
            f"Структура:\n- Функции: {', '.join(structure['functions']) or 'Нет'}\n- Классы: {', '.join(structure['classes']) or 'Нет'}\n"
            f"Ошибки:\n" + "\n".join(errors) + "\n"
            f"Рекомендации:\n{suggestions}\n"
            f"Точки интеграции:\n" + "\n".join([f"- {suggestion}" for _, suggestion in integration_points]) + "\n"
        )
        
        self.parent.display_response(inspection_text)

    def _save_code(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            messagebox.showwarning("Предупреждение", "Код пустой")
            return

        purpose, location = self.services.code_optimizer.classify_code(code)
        errors = self.services.code_optimizer.detect_errors(code)
        formatted_code = black.format_str(code, mode=black.FileMode()) if not errors else code

        code_id = uuid.uuid4().hex[:8]
        self.services.knowledge.save(f"Modified code (ID: {code_id})", formatted_code, context=f"Location: {location}, Purpose: {purpose}")
        
        self.parent.display_response(f"Код сохранен (ID: {code_id}):\n{formatted_code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}\n\nОшибки:\n" + "\n".join(errors))

    def _apply_to_app(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            messagebox.showwarning("Предупреждение", "Код пустой")
            return

        purpose, location = self.services.code_optimizer.classify_code(code)
        errors = self.services.code_optimizer.detect_errors(code)
        formatted_code = black.format_str(code, mode=black.FileMode()) if not errors else code

        module_name = "custom_module"
        with open(f"{module_name}.py", "w", encoding="utf-8") as f:
            f.write(formatted_code)
        
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        code_id = uuid.uuid4().hex[:8]
        self.services.knowledge.save(f"Applied code (ID: {code_id})", formatted_code, context=f"Location: {location}, Purpose: {purpose}")
        
        self.parent.display_response(f"Код применен к приложению (ID: {code_id}):\n{formatted_code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}\n\nОшибки:\n" + "\n".join(errors))
        self.destroy()

    def _duplicate_structure(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            messagebox.showwarning("Предупреждение", "Код пустой")
            return
        
        duplicated_code = self.services.code_optimizer.duplicate_structure(code)
        self.code_entry.delete("1.0", "end")
        self.code_entry.insert("1.0", duplicated_code)
        self._update_output()

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

            self.logo_label = ctk.CTkLabel(self, text="Nere More", font=("Arial", 20, "bold"), text_color="#FFFFFF")
            self.logo_label.pack(pady=10)

            self.input_frame = ctk.CTkFrame(self, fg_color="#2F3536", corner_radius=10)
            self.input_frame.pack(fill="x", padx=10, pady=5)
            self.input_entry = ctk.CTkEntry(self.input_frame, width=350, height=40, font=("Arial", 14),
                                            placeholder_text="Введите запрос...", fg_color="#1C2526", text_color="#FFFFFF")
            self.input_entry.pack(side="left", padx=10, pady=5)
            self.input_entry.bind("<Return>", lambda e: self.process_input())

            buttons = [
                ("📋", lambda: CodePasteWindow(self, self._paste_text_callback), "Вставить"),
                ("🧲", self._magnet_search, "Поиск"),
                ("🔍", self.process_input, "Обработать"),
                ("🌟", self.show_skills, "Умения"),
                ("⚙️", lambda: APISettingsWindow(self, self.config), "Настройки"),
                ("🔑", lambda: APIKeyCheckWindow(self, self.services, self.config), "Проверка ключа"),
                ("💻", lambda: CodeEditorWindow(self, self.services), "Редактор кода"),
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
        except Exception as e:
            logging.error(f"Критическая ошибка инициализации: {e}", exc_info=True)
            messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение: {e}")
            self.destroy()

    def _on_closing(self):
        logging.info("Закрытие приложения")
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
        self.display_response(f"Умения:\n- Генерация текста на основе опыта\n- Обработка кода\n- Работа с интернет-ресурсами\n- Обнаружение ошибок\n- Редактирование кода\n- Дублирование структуры\n- Инспекция кода\n\nТекущий уровень опыта: {self.services.knowledge.learning_rate:.1f}%")

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
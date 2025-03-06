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
from http.client import HTTPSConnection
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
from PIL import Image, ImageGrab
import pytesseract
import importlib.util
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("nere_more.log"), logging.StreamHandler()],
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def validate_folder_id(folder_id: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9]{20}$', folder_id))

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
            "learning": {"cluster_size": 5, "feedback_weight": 0.9},
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

    def analyze_from_screenshot(self, image_path: str) -> str:
        try:
            img = Image.open(image_path)
            code = pytesseract.image_to_string(img)
            if not code.strip():
                return "Не удалось извлечь код из изображения"
            return code
        except Exception as e:
            logging.error(f"Ошибка анализа скриншота: {e}")
            return f"Ошибка: {str(e)}"

class YandexGPT:
    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.url = "llm.api.cloud.yandex.net"
        self.path = "/foundationModels/v1/completion"
        self.model = "yandexgpt/latest"
        self.temperature = 0.68
        self.max_tokens = 500
        self.available = False
        self.status = "Не проверено"
        self.response_cache = OrderedDict(maxlen=1000)  # Кэш ответов
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
            conn = HTTPSConnection(self.url)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}".encode('latin-1')
            }
            payload = {
                "modelUri": f"gpt://{self.folder_id}/{self.model}",
                "completionOptions": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature
                },
                "messages": [{"role": "user", "text": "Test"}]
            }
            conn.request("POST", self.path, body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            self.available = response.status == 200
            self.status = "Доступно" if self.available else f"Ошибка: {response.status}"
            conn.close()
            return self.available, self.status
        except Exception as e:
            self.available = False
            self.status = f"Ошибка сети: {str(e)}"
            return self.available, self.status

    def invoke(self, query: str = None, context: str = "", json_payload: Dict[str, Any] = None) -> str:
        if not self.available:
            return "API недоступен"
        if not query and not json_payload:
            return "Запрос пустой"
        
        query_hash = hashlib.md5((query + context).encode()).hexdigest()
        if query_hash in self.response_cache:
            logging.info(f"Ответ взят из кэша для запроса: {query}")
            return self.response_cache[query_hash]

        try:
            conn = HTTPSConnection(self.url)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}".encode('latin-1')
            }
            
            payload = json_payload or {
                "modelUri": f"gpt://{self.folder_id}/{self.model}",
                "completionOptions": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature
                },
                "messages": [
                    {
                        "role": "system",
                        "text": "Предоставь четкий и понятный ответ в формате JSON с полями 'answer' и 'details'."
                    },
                    {
                        "role": "user",
                        "text": query if not context else f"{context}\n{query}"
                    }
                ]
            }
            
            conn.request("POST", self.path, body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            if response.status != 200:
                conn.close()
                return f"Ошибка: {response.status}"
            
            result = response.read().decode('utf-8')
            conn.close()
            
            try:
                json_result = json.loads(result)
                text = json_result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "Нет данных")
                json_response = json.loads(text)
                answer = json_response.get("answer", "Нет ответа")
                details = json_response.get("details", "Нет деталей")
                final_response = f"{answer}\n\nДетали: {details}"
                self.response_cache[query_hash] = final_response
                logging.info(f"Получен и закэширован ответ для запроса: {query}")
                return final_response
            except json.JSONDecodeError:
                logging.error(f"Ошибка парсинга JSON: {result}")
                return "Ошибка обработки ответа"
                
        except Exception as e:
            logging.error(f"Ошибка Yandex GPT: {e}")
            return "Ошибка сети"

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
                    embeddings = [np.frombuffer(entry['embeddings'], dtype=np.float16) for entry in self.db.all() if 'embeddings' in entry]
                    if embeddings:
                        self.kmeans.fit(np.stack(embeddings))
                    self._save_state()
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

    def save(self, query: str, response: str, context: str = "", feedback: float = 0.0):
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
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.db.insert(entry)

    def get_similar(self, query: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
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
        try:
            conn = HTTPSConnection(url.split('/')[2])
            conn.request("GET", '/' + '/'.join(url.split('/')[3:]))
            response = conn.getresponse()
            response.raise_for_status()
            soup = BeautifulSoup(response.read(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            conn.close()
            if text:
                self.save(query, text, context=f"Извлечено с сайта: {url}")
                return True
            return False
        except Exception as e:
            logging.error(f"Ошибка извлечения с {url}: {e}")
            return False

    def _clear_cache(self, text_id: str = None):
        with self._lock:
            if text_id:
                self.db.remove(Query().query.matches(f".*ID: {text_id}.*"))
            else:
                self.db.truncate()
                self.embeddings_cache.clear()
                self.query_cache.clear()
                self.vectorizer_fitted = False
                self.kmeans = None
            self._save_state()

class YandexAIServices:
    def __init__(self, gui_parent=None):
        self.config = Config()
        self.gui_parent = gui_parent
        self._request_credentials_if_needed()
        self.knowledge = KnowledgeBase(self)
        self.gpt = YandexGPT(self.config.get_key(), self.config.get_folder_id())
        
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

    def generate_response(self, query: str, context: str = "") -> str:
        if not query:
            return "Запрос пуст"
        urls = re.findall(r'https?://\S+', query)
        if urls:
            success = self.knowledge.save_web_content(urls[0], query)
            return f"Сохранено с {urls[0]}" if success else f"Ошибка с {urls[0]}"
        if "код" in query.lower() or "code" in query.lower():
            try:
                formatted_code = black.format_str(query, mode=black.FileMode())
                purpose, location = self.code_optimizer.classify_code(query)
                errors = self.code_optimizer.detect_errors(query)
                response = f"Код:\n{formatted_code}\n\nНазначение: {purpose}\nМесто: {location}\nОшибки: {', '.join(errors)}"
                return response
            except Exception as e:
                return f"Ошибка обработки кода: {e}"
        similar = self.knowledge.get_similar(query)
        return similar[0][1] if similar else self.gpt.invoke(query, context)

class NereMoreInterface(ctk.CTk):
    def __init__(self):
        logging.info("Начало инициализации интерфейса")
        try:
            super().__init__()
            self.title("Nere More")
            self.geometry("800x600")
            self.configure(fg_color="transparent")
            self.attributes('-alpha', 0.95)  # Прозрачность окна
            
            # Основной фрейм с полупрозрачным фоном
            self.main_frame = ctk.CTkFrame(self, fg_color="#1C2526", bg_color="#1C2526CC")
            self.main_frame.pack(fill="both", expand=True)

            self.audio = AudioManager()
            self.services = YandexAIServices(self)
            self.config = Config()
            self.context = deque(maxlen=self.config.data["ui"]["max_context"] * 2)

            # Заголовок
            self.logo_label = ctk.CTkLabel(self.main_frame, text="Nere More", 
                                         font=("Arial", 32, "bold"), 
                                         text_color="black")
            self.logo_label.pack(pady=20)

            # Поле ввода
            self.input_entry = ctk.CTkEntry(self.main_frame, width=600, height=50,
                                          font=("Arial", 20), 
                                          text_color="black",
                                          fg_color="white",
                                          placeholder_text="Введите запрос...")
            self.input_entry.pack(pady=10)
            self.input_entry.bind("<Return>", lambda e: self.process_input())

            # Кнопки
            self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
            self.button_frame.pack(pady=10)
            buttons = [
                ("Вставить", lambda: CodePasteWindow(self, self._paste_text_callback)),
                ("Поиск", self._magnet_search),
                ("Обработать", self.process_input),
                ("Умения", self.show_skills),
                ("Настройки", lambda: APISettingsWindow(self, self.config)),
            ]
            for text, cmd in buttons:
                btn = ctk.CTkButton(self.button_frame, text=text, width=100, height=40,
                                  font=("Arial", 16),
                                  text_color="black",
                                  fg_color="white",
                                  hover_color="#D3D3D3",
                                  command=cmd)
                btn.pack(side="left", padx=5)

            # Область результатов с эффектом падения
            self.results_text = ctk.CTkTextbox(self.main_frame, width=700, height=300,
                                             font=("Arial", 20),
                                             text_color="black",
                                             fg_color="transparent",
                                             bg_color="#1C2526CC")
            self.results_text.pack(pady=20)

            self.status_label = ctk.CTkLabel(self.main_frame, text="Готов", 
                                           font=("Arial", 16), 
                                           text_color="black")
            self.status_label.pack(pady=10)

            available, status = self.services.check_api_key()
            self.status_label.configure(text=f"Статус: {status}")

            self.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.initialized = True
        except Exception as e:
            logging.error(f"Ошибка инициализации: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось запустить: {e}")
            self.destroy()

    def _animate_text(self, text: str):
        self.results_text.delete("1.0", "end")
        lines = text.split('\n')
        for i, line in enumerate(lines):
            self.results_text.insert(f"{i+1}.0", line + '\n')
            self.results_text.update()
            time.sleep(0.05)  # Эффект падения текста

    def _on_closing(self):
        logging.info("Закрытие приложения")
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self.services.knowledge.db.close()
        self.destroy()

    def display_response(self, text: str):
        threading.Thread(target=self._animate_text, args=(text,), daemon=True).start()

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
            response = f"Код:\n{formatted_code}\n\nНазначение: {purpose}\nМесто: {location}\nОшибки: {', '.join(errors)}"
            self.services.knowledge.save(f"Inserted code (ID: {uuid.uuid4().hex[:8]})", formatted_code)
            self.display_response(response)
        else:
            self.services.knowledge.save(f"Inserted text (ID: {uuid.uuid4().hex[:8]})", content)
            self.display_response(f"Вставлено:\n{content}")

    def _magnet_search(self):
        query = self.input_entry.get().strip()
        if query:
            similar = self.services.knowledge.get_similar(query)
            response = "\n".join(f"[{s:.2f}] {q}:\n{r}" for q, r, s in similar) or "Нет данных"
            self.display_response(response)

    def show_skills(self):
        skills = "Умения:\n- Генерация текста\n- Обработка кода\n- Работа с файлами\n- Анализ скриншотов\n- Обнаружение ошибок"
        self.display_response(skills)

    def _get_context(self) -> str:
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.context)

    def run(self):
        if self.initialized:
            logging.info("Запуск приложения")
            self.mainloop()
        else:
            logging.error("Не инициализировано")

class CodePasteWindow(ctk.CTkToplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Вставка кода")
        self.geometry("400x300")
        self.configure(fg_color="transparent")
        self.callback = callback
        self._init_ui()

    def _init_ui(self):
        self.code_entry = ctk.CTkTextbox(self, width=380, height=200, 
                                       font=("Arial", 16),
                                       text_color="black",
                                       fg_color="white")
        self.code_entry.pack(padx=10, pady=10, fill="both", expand=True)
        self.code_entry.insert("1.0", "Вставьте код или текст здесь\n")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="Вставить", command=self._paste_code,
                     font=("Arial", 14), text_color="black", fg_color="white").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Отмена", command=self.destroy,
                     font=("Arial", 14), text_color="black", fg_color="white").pack(side="left", padx=5)

    def _paste_code(self):
        content = self.code_entry.get("1.0", "end-1c").strip()
        if content:
            self.callback(content)
        self.destroy()

class APISettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.title("Настройка API")
        self.geometry("300x250")
        self.configure(fg_color="transparent")
        self.config = config
        self._init_ui()

    def _init_ui(self):
        ctk.CTkLabel(self, text="YandexGPT key:", font=("Arial", 16), text_color="black").pack(pady=5)
        self.key_entry = ctk.CTkEntry(self, width=200, font=("Arial", 14), text_color="black", fg_color="white")
        self.key_entry.pack(pady=5)
        self.key_entry.insert(0, self.config.get_key())

        ctk.CTkLabel(self, text="Folder ID:", font=("Arial", 16), text_color="black").pack(pady=5)
        self.folder_entry = ctk.CTkEntry(self, width=200, font=("Arial", 14), text_color="black", fg_color="white")
        self.folder_entry.pack(pady=5)
        self.folder_entry.insert(0, self.config.get_folder_id())

        ctk.CTkButton(self, text="Сохранить", command=self._save_api_key,
                     font=("Arial", 14), text_color="black", fg_color="white").pack(pady=10)

        self.status_label = ctk.CTkLabel(self, text="", font=("Arial", 14), text_color="black")
        self.status_label.pack(pady=5)

    def _save_api_key(self):
        key = self.key_entry.get().strip()
        folder_id = self.folder_entry.get().strip()

        if not key or not folder_id:
            self.status_label.configure(text="Поля не могут быть пустыми")
            return

        if not validate_folder_id(folder_id):
            self.status_label.configure(text="folder_id должен быть 20 символов")
            return

        temp_gpt = YandexGPT(key, folder_id)
        is_valid, status_message = temp_gpt.check_availability()
        self.status_label.configure(text=status_message)

        if is_valid:
            self.config.update_api_key("gpt_key_1", key)
            self.config.update_folder_id(folder_id)
            self.status_label.configure(text="Сохранено")
            self.after(1000, self.destroy)

if __name__ == "__main__":
    app = NereMoreInterface()
    app.run()
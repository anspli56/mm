from collections import deque
import json
import logging
import hashlib
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
from typing import Dict, Any, List, Tuple
import re
from cryptography.fernet import Fernet
import black
from pylint import lint
from io import StringIO
from pylint.reporters.text import TextReporter
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
                "iam_token": "",
                "folder_id": ""
            },
            "ui": {"animation_speed": 0.05, "max_context": 10, "audio_enabled": True},
            "learning": {"feedback_weight": 0.9}
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
        iam_token = self.get_iam_token()
        folder_id = self.get_folder_id()
        
        if not iam_token or not folder_id:
            logging.info("IAM-токен или folder_id отсутствуют, используются значения по умолчанию")
            self.update_iam_token("default_iam_token")
            self.update_folder_id("b1g170pkl3ihbn8bc3kd")
            return
            
        temp_gpt = YandexGPT(iam_token, folder_id)
        available, status = temp_gpt.check_availability()
        
        if not available:
            logging.warning(f"Сохраненные данные недействительны: {status}. Установка значений по умолчанию")
            self.update_iam_token("default_iam_token")
            self.update_folder_id("b1g170pkl3ihbn8bc3kd")

    @property
    def data(self) -> Dict[str, Any]:
        return self.config

    def get_iam_token(self) -> str:
        try:
            return self._cipher.decrypt(self.config["yandex"]["iam_token"].encode()).decode()
        except Exception:
            return self.config["yandex"]["iam_token"]

    def update_iam_token(self, value: str) -> bool:
        temp_gpt = YandexGPT(value, self.get_folder_id())
        available, status = temp_gpt.check_availability()
        
        if not available:
            logging.error(f"Новый IAM-токен недействителен: {status}")
            return False
            
        encrypted_value = self._cipher.encrypt(value.encode()).decode()
        self.config["yandex"]["iam_token"] = encrypted_value
        self._save_config()
        return True

    def get_folder_id(self) -> str:
        return self.config["yandex"].get("folder_id", "")

    def update_folder_id(self, folder_id: str) -> bool:
        if not validate_folder_id(folder_id):
            logging.error("Недействительный folder_id")
            return False
            
        iam_token = self.get_iam_token()
        if iam_token:
            temp_gpt = YandexGPT(iam_token, folder_id)
            available, status = temp_gpt.check_availability()
            if not available:
                logging.error(f"folder_id недействителен с текущим токеном: {status}")
                return False
                
        self.config["yandex"]["folder_id"] = folder_id
        self._save_config()
        return True

class CodeOptimizationModule:
    def __init__(self, config: Config):
        self.config = config.data.get("code_classification", {
            "purposes": ["algorithm", "UI", "data_processing", "network", "utility"],
            "locations": ["core", "GUI", "services", "knowledge"],
            "keywords": {
                "algorithm": ["def", "for", "while", "if"],
                "UI": ["tkinter", "ctk", "label", "button"],
                "data_processing": ["numpy", "sklearn", "pandas"],
                "network": ["requests", "socket"],
                "utility": ["os", "logging", "time"]
            }
        })

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
        
        try:
            output = StringIO()
            reporter = TextReporter(output)
            args = ["--from-stdin", "--persistent=n", "--score=n"]
            pylint_run = lint.Run(args, reporter=reporter, do_exit=False)
            pylint_run.linter.check_single_file("stdin", code.splitlines())
            pylint_output = output.getvalue()
            output.close()
            
            for line in pylint_output.splitlines():
                if "error" in line.lower() or "warning" in line.lower():
                    errors.append(line.strip())
        except Exception as e:
            errors.append(f"Ошибка анализа Pylint: {str(e)}")
        
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
            if "missing" in error.lower() and "docstring" in error.lower():
                suggestions.append("Добавьте docstring для функций/классов.")
            elif "undefined" in error.lower():
                suggestions.append("Проверьте объявление переменных перед использованием.")
            elif "syntax" in error.lower():
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
    def __init__(self, iam_token: str, folder_id: str):
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.url = "llm.api.cloud.yandex.net"
        self.path = "/foundationModels/v1/completion"
        self.model = "yandexgpt/latest"
        self.temperature = 0.68
        self.max_tokens = 500
        self.available = False
        self.status = "Не проверено"
        self._validate_credentials()

    def _validate_credentials(self):
        if not self.iam_token or len(self.iam_token.strip()) < 10:
            self.status = "Ошибка: IAM-токен пустой или слишком короткий"
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
                "Authorization": f"Bearer {self.iam_token}",
                "x-folder-id": self.folder_id
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
            return f"API отключен: {self.status}"
        if not query and not json_payload:
            return "Ошибка: Запрос пустой"
        
        try:
            conn = HTTPSConnection(self.url)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.iam_token}",
                "x-folder-id": self.folder_id
            }
            
            payload = json_payload or {
                "modelUri": f"gpt://{self.folder_id}/{self.model}",
                "completionOptions": {
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature
                },
                "messages": [
                    {"role": "system", "text": "Ты логичный и полезный помощник. Давай чёткие и понятные ответы."},
                    {"role": "user", "text": query if not context else f"{context}\n{query}"}
                ]
            }
            
            conn.request("POST", self.path, body=json.dumps(payload), headers=headers)
            response = conn.getresponse()
            if response.status != 200:
                conn.close()
                return f"Ошибка: {response.status} {response.reason}"
            
            result = response.read().decode('utf-8')
            conn.close()
            
            json_result = json.loads(result)
            return json_result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "No data")
        except Exception as e:
            logging.error(f"Ошибка Yandex GPT: {e}")
            return f"Ошибка сети: {str(e)}"

class KnowledgeBase:
    def __init__(self, services: 'YandexAIServices'):
        self.services = services
        self.db = TinyDB("nere_more_knowledge.json")
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self._lock = threading.Lock()
        self.query_cache = {}
        self.vectorizer_fitted = False
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        docs = [entry['response'] for entry in self.db.all() if 'response' in entry]
        if docs:
            self.vectorizer.fit(docs)
            self.vectorizer_fitted = True

    def save(self, query: str, response: str, context: str = ""):
        with self._lock:
            if len(response.strip()) < 5:
                return
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.query_cache[query_hash] = response
            if not self.vectorizer_fitted:
                self.vectorizer.fit([response])
                self.vectorizer_fitted = True
            embeddings = self.vectorizer.transform([response]).toarray().astype(np.float16)[0]
            entry = {
                'id': str(uuid.uuid4()),
                'query': query,
                'response': response,
                'context': context,
                'embeddings': embeddings.tobytes(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.db.insert(entry)

    def get_similar(self, query: str, top_n: int = 1) -> List[Tuple[str, str, float]]:
        if not self.vectorizer_fitted:
            return []
        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float16)[0]
        all_data = [(entry['query'], entry['response'], np.frombuffer(entry['embeddings'], dtype=np.float16))
                    for entry in self.db.all() if 'embeddings' in entry]
        if not all_data:
            return []
        embeddings = np.stack([item[2] for item in all_data])
        similarities = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
        results = sorted(zip(all_data, similarities), key=lambda x: x[1], reverse=True)[:top_n]
        return [(q, r, float(s)) for (q, r, _), s in results]

    def _clear_cache(self, text_id: str = None):
        with self._lock:
            if text_id:
                self.db.remove(Query().query.matches(f".*ID: {text_id}.*"))
            else:
                self.db.truncate()
                self.query_cache.clear()
                self.vectorizer_fitted = False
            self._save_state()

    def _save_state(self):
        pass  # Убрано сохранение моделей для упрощения

class YandexAIServices:
    def __init__(self, gui_parent=None):
        self.config = Config()
        self.gui_parent = gui_parent
        self._request_credentials_if_needed()
        self.knowledge = KnowledgeBase(self)
        self.gpt = YandexGPT(self.config.get_iam_token(), self.config.get_folder_id())
        self.code_optimizer = CodeOptimizationModule(self.config)
        self.learning_progress = {"queries_processed": 0, "average_response_time": 0.0}
        self.last_interaction = time.time()
        
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
        if not self.config.get_iam_token():
            iam_token = ctk.CTkInputDialog(text="Введите IAM-токен:", title="IAM Token").get_input()
            if iam_token and self.config.update_iam_token(iam_token):
                logging.info("IAM-токен обновлен")

    def check_api_key(self) -> Tuple[bool, str]:
        return self.gpt.check_availability()

    def generate_response(self, query: str, context: str = "") -> str:
        start_time = time.time()
        
        if not query and (time.time() - self.last_interaction) > 30:
            return "Ты давно молчишь! Давай я предложу идеи: хочешь обсудить код, проекты или что-то ещё?"
        
        self.last_interaction = time.time()
        
        if not query:
            return "Что ты хочешь обсудить? Я готов помочь!"
        
        similar = self.knowledge.get_similar(query)
        if similar and similar[0][2] > 0.9:
            response = similar[0][1]
        else:
            response = self.gpt.invoke(query, context)
            self.knowledge.save(query, response, context)
        
        self.learning_progress["queries_processed"] += 1
        self.learning_progress["average_response_time"] = (
            (self.learning_progress["average_response_time"] * (self.learning_progress["queries_processed"] - 1)) + 
            (time.time() - start_time)) / self.learning_progress["queries_processed"]
        )
        
        if "что ты хочешь" in query.lower():
            response += "\nЯ могу помочь с кодом, анализом данных или просто поболтать. Что тебе интересно?"
        
        return response

    def get_learning_progress(self) -> str:
        return (f"Прогресс обучения:\n"
                f"- Обработано запросов: {self.learning_progress['queries_processed']}\n"
                f"- Среднее время ответа: {self.learning_progress['average_response_time']:.2f} сек")

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
        self.formatted_output = tk.Text(self.output_frame, height=10, width=40, bg="#1C2526", fg="#FFFFFF", font=("Courier", 12))
        self.formatted_output.pack(fill="both", expand=True, pady=5)

        self.changes_label = ctk.CTkLabel(self.output_frame, text="Изменения и интеграция:", text_color="#FFFFFF")
        self.changes_label.pack(pady=2)
        self.changes_output = tk.Text(self.output_frame, height=10, width=40, bg="#1C2526", fg="#FFFFFF", font=("Courier", 12))
        self.changes_output.pack(fill="both", expand=True, pady=5)

        button_frame = ctk.CTkFrame(self.main_frame, fg_color="#2F3536")
        button_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(button_frame, text="Сохранить", command=self._save_code, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Проверить", command=self._inspect_code, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Запустить", command=self._run_code, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Применить", command=self._apply_to_app, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Дублировать", command=self._duplicate_structure, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Скриншот", command=self._take_screenshot, fg_color="#1C2526", hover_color="#4A4A4A").pack(side="left", padx=5)
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
            self.parent.display_response("Ошибка: Код пустой")
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
            self.parent.display_response("Ошибка: Код пустой")
            return

        purpose, location = self.services.code_optimizer.classify_code(code)
        errors = self.services.code_optimizer.detect_errors(code)
        formatted_code = black.format_str(code, mode=black.FileMode()) if not errors else code

        code_id = uuid.uuid4().hex[:8]
        self.services.knowledge.save(f"Modified code (ID: {code_id})", formatted_code, context=f"Location: {location}, Purpose: {purpose}")
        
        self.parent.display_response(f"Код сохранен (ID: {code_id}):\n{formatted_code}\n\nКлассификация:\n- Назначение: {purpose}\n- Место: {location}\n\nОшибки:\n" + "\n".join(errors))

    def _run_code(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            self.parent.display_response("Ошибка: Код пустой")
            return

        sandbox_globals = {}
        try:
            exec(code, sandbox_globals)
            self.parent.display_response("Код успешно выполнен в песочнице")
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.parent.display_response(f"Ошибка выполнения:\n{stack_trace}")

    def _apply_to_app(self):
        code = self.code_entry.get("1.0", "end-1c").strip()
        if not code:
            self.parent.display_response("Ошибка: Код пустой")
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
            self.parent.display_response("Ошибка: Код пустой")
            return
        
        duplicated_code = self.services.code_optimizer.duplicate_structure(code)
        self.code_entry.delete("1.0", "end")
        self.code_entry.insert("1.0", duplicated_code)
        self._update_output()

    def _take_screenshot(self):
        try:
            screenshot = ImageGrab.grab(bbox=(self.winfo_x(), self.winfo_y(), self.winfo_x() + self.winfo_width(), self.winfo_y() + self.winfo_height()))
            filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
            screenshot.save(filename)
            self.parent.display_response(f"Скриншот сохранен как {filename}")
        except Exception as e:
            self.parent.display_response(f"Ошибка создания скриншота: {e}")

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
            self.status_label.configure(text="Готов")
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

    def process_input(self):
        query = self.input_entry.get().strip()
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
        progress = self.services.get_learning_progress()
        self.display_response(f"Умения:\n- Генерация текста\n- Обработка кода\n- Работа с файлами\n- Анализ скриншотов\n- Обнаружение ошибок\n- Редактирование кода\n- Динамическая перезагрузка\n- Дублирование структуры\n- Инспекция кода\n\n{progress}")

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
        ctk.CTkLabel(self, text="IAM Token:").grid(row=0, column=0, padx=5, pady=5)
        self.key_entry = ctk.CTkEntry(self, width=150)
        self.key_entry.grid(row=0, column=1, padx=5, pady=5)
        self.key_entry.insert(0, self.config.get_iam_token())

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
            self.config.update_iam_token(key)
            self.config.update_folder_id(folder_id)
            self.status_label.configure(text="Настройки сохранены")
            self.after(1000, self.destroy)
        else:
            self.status_label.configure(text=f"Ошибка: {status_message}")

class APIKeyCheckWindow(ctk.CTkToplevel):
    def __init__(self, parent, services, config):
        super().__init__(parent)
        self.title("Проверка IAM-токена")
        self.geometry("400x300")
        self.services = services
        self.config = config
        self._init_ui()

    def _init_ui(self):
        ctk.CTkLabel(self, text="Введите токен:").grid(row=0, column=0, padx=5, pady=5)
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
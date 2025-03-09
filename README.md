import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import asyncio
import json
import logging
import sqlite3
import threading
import ast
import re
from functools import lru_cache, wraps
from collections import deque, OrderedDict
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydantic import BaseModel, ValidationError
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.mixed_precision import set_global_policy
import aiohttp
import requests
from typing import Dict, Any, List, Tuple, Optional
import time
import uuid
import black
import random
from cryptography.fernet import Fernet
from bs4 import BeautifulSoup

# Placeholder for Horovod (requires MPI setup)
# import horovod.tensorflow as hvd

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("nere_more.log"), logging.StreamHandler()],
)

# Enable mixed precision
set_global_policy('mixed_float16')

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Error handling decorator
def log_errors(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Ошибка в {func.__name__}: {e}", exc_info=True)
            raise
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Ошибка в {func.__name__}: {e}", exc_info=True)
            raise
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Sentiment Analyzer with BERT
class SentimentAnalyzer:
    def __init__(self, model_name="blanchefort/rubert-base-cased-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.mood_history = deque(maxlen=50)
        logging.info(f"SentimentAnalyzer initialized with {model_name}")

    @lru_cache(maxsize=100)
    def predict_sentiment_score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        result = self.sentiment_pipeline(text)[0]
        label, score = result['label'], result['score']
        sentiment = score if label == 'POSITIVE' else -score if label == 'NEGATIVE' else 0.0
        self.mood_history.append(sentiment)
        return sentiment

    def analyze_mood_trend(self) -> Dict[str, Any]:
        if not self.mood_history:
            return {"current_mood": 0.0, "average_mood": 0.0, "trend": "нет данных"}
        current_mood = self.mood_history[-1]
        average_mood = sum(self.mood_history) / len(self.mood_history)
        trend = "недостаточно данных"
        if len(self.mood_history) >= 2:
            recent = list(self.mood_history)[-5:]
            trend = "рост" if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)) else "спад" if all(recent[i] >= recent[i+1] for i in range(len(recent)-1)) else "стабильное"
        return {"current_mood": current_mood, "average_mood": average_mood, "trend": trend}

    def interpret_mood(self, mood_score: float) -> str:
        return "позитивное" if mood_score > 0.5 else "негативное" if mood_score < -0.5 else "нейтральное"

# Input validation with Pydantic
class UserInput(BaseModel):
    query: str
    context: Optional[str] = ""

def validate_folder_id(folder_id: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9]{20}$', folder_id))

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# SQLite-based Knowledge Base
class KnowledgeBase:
    def __init__(self, services: 'YandexAIServices'):
        self.services = services
        self.conn = sqlite3.connect("knowledge.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge
                            (id TEXT PRIMARY KEY, query TEXT, response TEXT, context TEXT, timestamp TEXT, experience REAL)''')
        self.conn.commit()
        self.lock = threading.Lock()
        self.context_history = deque(maxlen=50)
        self.experience_level = self._get_experience_level()

    def _get_experience_level(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM knowledge")
        return self.cursor.fetchone()[0]

    def update_experience(self) -> float:
        self.experience_level += 1
        fib_value = fibonacci(min(self.experience_level, 20))
        learning_rate = min(100.0, fib_value * 0.5)
        return learning_rate

    def save(self, query: str, response: str, context: str = ""):
        with self.lock:
            if len(response.strip()) < 5:
                return
            entry = (str(uuid.uuid4()), query, response, context, time.strftime('%Y-%m-%d %H:%M:%S'), self.update_experience())
            self.cursor.execute("INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?)", entry)
            self.conn.commit()
            self.context_history.append(f"Вопрос: {query}\nОтвет: {response}")

    def get_similar(self, query: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        self.cursor.execute("SELECT query, response FROM knowledge")
        data = self.cursor.fetchall()
        similarities = [(q, r, 1.0 if query.lower() in q.lower() else 0.5) for q, r in data]
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_n]

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
        similar = self.get_similar(query, 3)
        context = "\n\n".join([f"Ранее: {q}\nОтвет: {r} (сходство: {s:.2f})" for q, r, s in similar]) or "Нет схожих данных"
        history = "\n".join(list(self.context_history)[-5:])
        return f"История:\n{history}\n\nПрошлый опыт:\n{context}"

# Asynchronous YandexGPT
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

    @log_errors
    async def check_availability(self) -> Tuple[bool, str]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Api-Key {self.api_key}", "Content-Type": "application/json"}
            payload = {"modelUri": f"gpt://{self.folder_id}/{self.model}", "messages": [{"role": "user", "text": "Test"}]}
            async with session.post(self.url, headers=headers, json=payload) as resp:
                self.available = resp.status == 200
                self.status = "Доступно" if self.available else f"Ошибка: {resp.status}"
                return self.available, self.status

    @log_errors
    async def invoke(self, json_payload: Dict[str, Any]) -> str:
        if not self.available:
            return f"API отключен: {self.status}"
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Api-Key {self.api_key}", "Content-Type": "application/json"}
            async with session.post(self.url, headers=headers, json=json_payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "No data")

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
            return cls._instance

    def _load_config(self):
        self.default_config = {
            "yandex": {"keys": [{"id": "gpt_key_1", "value": "", "type": "gpt"}], "folder_id": ""},
            "ui": {"animation_speed": 0.05, "max_context": 10}
        }
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                    self.config = {**self.default_config, **self.config}
            else:
                self.config = self.default_config
                self._save_config()
        except Exception as e:
            logging.error(f"Ошибка загрузки конфигурации: {e}")
            self.config = self.default_config

    def _save_config(self):
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def get_key(self) -> str:
        for key in self.config["yandex"]["keys"]:
            if key["type"] == "gpt":
                try:
                    return self._cipher.decrypt(key["value"].encode()).decode()
                except Exception:
                    return key["value"]
        return ""

    def update_api_key(self, key_id: str, value: str) -> bool:
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
            return False
        self.config["yandex"]["folder_id"] = folder_id
        self._save_config()
        return True

class CodeOptimizationModule:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.config = {"purposes": ["algorithm", "UI", "data", "network", "utility"],
                      "keywords": {"algorithm": ["def", "for"], "UI": ["tkinter", "ctk"],
                                   "data": ["numpy", "pandas"], "network": ["requests"], "utility": ["os"]}}

    def classify_code(self, code: str) -> Tuple[str, str]:
        purpose_score = {p: 0 for p in self.config["purposes"]}
        tokens = re.findall(r'\w+', code.lower())
        for purpose, keywords in self.config["keywords"].items():
            for keyword in keywords:
                if keyword in tokens:
                    purpose_score[purpose] += 1
        purpose = max(purpose_score, key=purpose_score.get, default="utility")
        location = "GUI" if "ctk" in code else "core"
        return purpose, location

    def detect_errors(self, code: str) -> List[str]:
        try:
            ast.parse(code)
            return ["Ошибок не обнаружено"]
        except SyntaxError as e:
            return [f"Синтаксическая ошибка: {str(e)}"]

    def analyze_comments_with_flax(self, code: str) -> Dict[str, float]:
        comments = [line[line.find('#')+1:].strip() for line in code.split('\n') if '#' in line and line.strip().startswith('#')]
        return {comment: self.sentiment_analyzer.predict_sentiment_score(comment) for comment in comments if comment}

# Main Service Class
class YandexAIServices:
    def __init__(self, gui_parent=None):
        self.config = Config()
        self.gui_parent = gui_parent
        self.gpt = YandexGPT(self.config.get_key(), self.config.get_folder_id())
        self.knowledge = KnowledgeBase(self)
        self.code_optimizer = CodeOptimizationModule()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.loop = asyncio.get_event_loop()
        self.metrics = {"requests": 0, "avg_response_time": 0.0}
        self.loop.run_until_complete(self.initialize())

    async def initialize(self):
        available, status = await self.gpt.check_availability()
        if not available and self.gui_parent:
            self.gui_parent.status_label.configure(text=f"Ошибка API: {status}")

    @log_errors
    async def generate_response(self, query: str, context: str = "") -> str:
        try:
            validated = UserInput(query=query, context=context)
        except ValidationError as e:
            return f"Ошибка валидации: {e}"
        
        start_time = time.time()
        self.metrics["requests"] += 1
        urls = re.findall(r'https?://\S+', query)
        if urls:
            success = self.knowledge.save_web_content(urls[0], query)
            return f"Сохранено с {urls[0]}" if success else f"Ошибка с {urls[0]}"
        
        if "код" in query.lower() or "code" in query.lower():
            purpose, location = self.code_optimizer.classify_code(query)
            errors = self.code_optimizer.detect_errors(query)
            formatted_code = black.format_str(query, mode=black.FileMode()) if "Ошибка" not in errors[0] else query
            sentiment_scores = self.code_optimizer.analyze_comments_with_flax(query)
            comment_analysis = "\n".join([f"- '{c}' -> {self.sentiment_analyzer.interpret_mood(s)} ({s:.2f})" for c, s in sentiment_scores.items()]) or "Нет комментариев"
            response = f"Код:\n{formatted_code}\nКлассификация: {purpose}, {location}\nОшибки: {chr(10).join(errors)}\nКомментарии:\n{comment_analysis}"
            self.knowledge.save(query, response)
            return response
        
        user_emotion = self.sentiment_analyzer.predict_sentiment_score(query)
        mood_analysis = self.sentiment_analyzer.analyze_mood_trend()
        tone = "радостный" if user_emotion > 0.5 else "поддерживающий" if user_emotion < -0.5 else "нейтральный"
        built_context = self.knowledge.build_context(query)
        prompt = {
            "modelUri": f"gpt://{self.gpt.folder_id}/{self.gpt.model}",
            "completionOptions": {"temperature": 0.5, "maxTokens": 2000},
            "messages": [
                {"role": "system", "text": f"Тон: {tone}, Опыт: {self.knowledge.experience_level}"},
                {"role": "user", "text": f"Контекст: {built_context}\nЗапрос: {query}"}
            ]
        }
        response = await self.gpt.invoke(prompt)
        resp_score = self.sentiment_analyzer.predict_sentiment_score(response)
        self.knowledge.save(query, response, built_context)
        elapsed = time.time() - start_time
        self.metrics["avg_response_time"] = (self.metrics["avg_response_time"] * (self.metrics["requests"] - 1) + elapsed) / self.metrics["requests"]
        
        return (f"Ответ: {response}\n"
                f"Настроение: {self.sentiment_analyzer.interpret_mood(user_emotion)} ({user_emotion:.2f})\n"
                f"Оценка ответа: {resp_score:.2f}\n"
                f"Тренд: {mood_analysis['trend']}\n"
                f"Метрики: Запросов: {self.metrics['requests']}, Время: {self.metrics['avg_response_time']:.2f}с")

class InteractiveBehavior:
    def __init__(self, gui_interface):
        self.gui = gui_interface
        self.last_interaction = time.time()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.greetings = ["Привет!", "Здорово!", "Как дела?"]
        self.questions = ["Чем занимаешься?", "Что нового?"]
        self.suggestions = ["Давай проанализируем код?", "Хочешь обучить модель?"]
        self.is_running = False
        self.thread = None

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
            if time.time() - self.last_interaction > 30:
                self._interact_with_user()
                self.last_interaction = time.time()
            time.sleep(5)

    def _interact_with_user(self):
        last_input = self.gui.input_entry.get().strip()
        if last_input:
            score = self.sentiment_analyzer.predict_sentiment_score(last_input)
            greeting = random.choice(self.greetings) + (" Ты счастлив!" if score > 0.5 else " Не грусти!" if score < -0.5 else "")
            suggestion = random.choice(self.suggestions)
            self.gui.display_response(f"{greeting}\n{suggestion}")

    def update_last_interaction(self):
        self.last_interaction = time.time()

class NereMoreInterface(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Nere More")
        self.geometry("600x450")
        self.configure(fg_color="#1C2526")
        self.services = YandexAIServices(self)
        self.loop = asyncio.get_event_loop()
        self.interactive_behavior = InteractiveBehavior(self)
        
        self.input_entry = ctk.CTkEntry(self, width=350, placeholder_text="Введите запрос...")
        self.input_entry.pack(pady=5)
        self.input_entry.bind("<Return>", lambda e: self.process_input())
        
        self.results_text = ctk.CTkTextbox(self, width=580, height=200)
        self.results_text.pack(pady=5)
        
        self.plot_button = ctk.CTkButton(self, text="Показать настроение", command=self.plot_mood)
        self.plot_button.pack(pady=5)
        
        self.status_label = ctk.CTkLabel(self, text="Готов")
        self.status_label.pack(pady=2)
        
        self.interactive_behavior.start()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.interactive_behavior.stop()
        self.services.knowledge.conn.close()
        self.destroy()

    def display_response(self, text: str):
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)

    def process_input(self):
        self.interactive_behavior.update_last_interaction()
        query = self.input_entry.get().strip()
        if query:
            response = self.loop.run_until_complete(self.services.generate_response(query))
            self.display_response(response)

    def plot_mood(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        mood_history = list(self.services.sentiment_analyzer.mood_history)
        ax.plot(mood_history, label="Настроение")
        ax.set_title("Тенденция настроения")
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=5)

if __name__ == "__main__":
    app = NereMoreInterface()
    app.mainloop()
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Отключаем oneDNN для устранения сообщений TensorFlow

import asyncio
from collections import deque
import json
import logging
import hashlib
import customtkinter as ctk
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
import black
import requests
from tinydb import TinyDB, Query
import joblib
from cryptography.fernet import Fernet
import pandas as pd
from fastai.text.all import TextDataLoaders, AWD_LSTM, text_learner, mae, MSELossFlat, load_learner
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
from jax import random
import optax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("nere_more.log"), logging.StreamHandler()],
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Оптимизированный Fast.ai
class FastAITextAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(*args, **kwargs)
            return cls._instance

    def _initialize(self, csv_path="sentiment_data.csv", model_path="text_regressor.pth"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.learn = None
        self.mood_history = deque(maxlen=50)
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.learn = load_learner(self.model_path)
                logging.info("Загружена сохраненная модель Fast.ai")
            else:
                self._train_model()
            if self.learn:
                torch.cuda.empty_cache()  # Очистка GPU-памяти, если используется
        except Exception as e:
            logging.error(f"Ошибка загрузки Fast.ai модели: {e}")
            self.learn = None

    def _train_model(self):
        try:
            data = self._load_or_create_data()
            dls = TextDataLoaders.from_df(
                data, text_col='text', label_col='score', valid_pct=0.2, bs=16, seq_len=72
            )
            self.learn = text_learner(
                dls, AWD_LSTM, drop_mult=0.3, metrics=[mae], loss_func=MSELossFlat()
            )
            self.learn.fit_one_cycle(3, 1e-2)
            self.learn.export(self.model_path)
            logging.info("Модель Fast.ai обучена и сохранена")
        except Exception as e:
            logging.error(f"Ошибка обучения Fast.ai: {e}")
            self.learn = None

    def _load_or_create_data(self):
        try:
            if os.path.exists(self.csv_path):
                data = pd.read_csv(self.csv_path)
                if 'text' not in data.columns or 'score' not in data.columns:
                    raise ValueError("Неверный формат CSV")
                return data
            return self._create_default_dataset()
        except Exception as e:
            logging.warning(f"Ошибка загрузки данных: {e}. Создается стандартный набор.")
            return self._create_default_dataset()

    def _create_default_dataset(self):
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
        return data

    async def predict_sentiment_score(self, text: str) -> float:
        if not self.learn:
            logging.warning("Fast.ai модель недоступна, используется нейтральная оценка")
            return 0.0
        try:
            pred = await asyncio.to_thread(self.learn.predict, text)
            score = float(pred[0].item())
            self.mood_history.append((score, time.time()))  # Сохраняем с временной меткой
            return score
        except Exception as e:
            logging.error(f"Ошибка предсказания Fast.ai: {e}")
            return 0.0

    def analyze_mood_trend(self) -> Dict[str, Any]:
        if not self.mood_history:
            return {"current_mood": 0.0, "average_mood": 0.0, "trend": "нет данных"}
        
        current_mood, current_time = self.mood_history[-1]
        recent_moods = [(score, t) for score, t in self.mood_history if current_time - t <= 3600]  # Последний час
        if not recent_moods:
            return {"current_mood": current_mood, "average_mood": 0.0, "trend": "нет данных"}
        
        scores = [s for s, _ in recent_moods]
        average_mood = sum(scores) / len(scores)
        trend = "стабильное настроение"
        if len(scores) >= 3:
            slope = np.polyfit(range(len(scores)), scores, 1)[0]
            trend = "рост настроения" if slope > 0.01 else "спад настроения" if slope < -0.01 else "стабильное настроение"
        
        return {"current_mood": current_mood, "average_mood": average_mood, "trend": trend}

    def interpret_mood(self, score: float) -> str:
        if score > 0.5: return "позитивное"
        elif score < -0.5: return "негативное"
        return "нейтральное"

# Оптимизированный анализ кода
class CodeOptimizationModule:
    def __init__(self, config: Config):
        self.config = config.data["code_classification"]
        self.text_analyzer = FastAITextAnalyzer()
        self.code_cache = {}  # Кэш для результатов анализа

    async def classify_code(self, code: str) -> Tuple[str, str]:
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash in self.code_cache:
            return self.code_cache[code_hash]["classification"]
        
        try:
            tree = ast.parse(code)
            purposes = {"algorithm": 0, "UI": 0, "data_processing": 0, "network": 0, "utility": 0}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    purposes["algorithm"] += 1
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    module = node.names[0].name if isinstance(node, ast.Import) else node.module
                    if module in ["tkinter", "customtkinter"]: purposes["UI"] += 2
                    elif module in ["numpy", "pandas", "sklearn"]: purposes["data_processing"] += 2
                    elif module in ["requests", "socket"]: purposes["network"] += 2
                    elif module in ["os", "sys", "logging"]: purposes["utility"] += 1
            
            purpose = max(purposes, key=purposes.get, default="utility")
            location = "core"
            if purposes["UI"] > 0: location = "GUI"
            elif purposes["network"] > 0: location = "services"
            elif purposes["data_processing"] > 0: location = "knowledge"
            
            self.code_cache[code_hash] = {"classification": (purpose, location)}
            return purpose, location
        except Exception as e:
            logging.error(f"Ошибка классификации кода: {e}")
            return "utility", "core"

    async def detect_errors(self, code: str) -> List[str]:
        try:
            ast.parse(code)
            return ["Ошибок не обнаружено"]
        except SyntaxError as e:
            return [f"Синтаксическая ошибка: {str(e)}"]
        except Exception as e:
            return [f"Ошибка анализа: {str(e)}"]

    async def analyze_comments(self, code: str) -> Dict[str, float]:
        comments = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                comment = line[1:].strip()
                if comment: comments.append(comment)
            elif '#' in line:
                comment = line[line.find('#')+1:].strip()
                if comment: comments.append(comment)
        
        if not comments:
            return {}
        
        sentiment_scores = {}
        for comment in comments:
            score = await self.text_analyzer.predict_sentiment_score(comment)
            sentiment_scores[comment] = score
        return sentiment_scores

    async def suggest_structure(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            suggestions = []
            func_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            
            if func_count > 5:
                suggestions.append("Рекомендация: разбейте код на модули, функций слишком много.")
            if class_count > 1:
                suggestions.append("Рекомендация: проверьте, можно ли объединить классы или вынести в отдельные файлы.")
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.body:
                    suggestions.append(f"Функция '{node.name}' пуста, добавьте реализацию или удалите.")
                elif isinstance(node, ast.FunctionDef) and not ast.get_docstring(node):
                    suggestions.append(f"Добавьте документацию к функции '{node.name}'.")
            
            return "\n".join(suggestions) or "Структура кода выглядит хорошо."
        except Exception as e:
            return f"Ошибка анализа структуры: {str(e)}"

# Основной сервис
class YandexAIServices:
    def __init__(self, gui_parent=None):
        self.config = Config()
        self.gui_parent = gui_parent
        self.knowledge = KnowledgeBase(self)
        self.gpt = YandexGPT(self.config.get_key(), self.config.get_folder_id())
        self.code_optimizer = CodeOptimizationModule(self.config)
        self.text_analyzer = FastAITextAnalyzer()
        self.loop = asyncio.get_event_loop()

    async def generate_response(self, query: str) -> str:
        if not query:
            return "Ошибка: Запрос пуст"
        
        user_emotion = await self.text_analyzer.predict_sentiment_score(query)
        mood_analysis = self.text_analyzer.analyze_mood_trend()
        mood_summary = (
            f"Анализ настроения:\n"
            f"- Текущий: {self.text_analyzer.interpret_mood(user_emotion)} ({user_emotion:.2f})\n"
            f"- Средний: {self.text_analyzer.interpret_mood(mood_analysis['average_mood'])} ({mood_analysis['average_mood']:.2f})\n"
            f"- Тенденция: {mood_analysis['trend']}"
        )

        if "код" in query.lower() or "code" in query.lower():
            return await self._handle_code(query, mood_summary)
        
        urls = re.findall(r'https?://\S+', query)
        if urls:
            success = self.knowledge.save_web_content(urls[0], query)
            return f"{'Сохранено с ' + urls[0] if success else 'Ошибка с ' + urls[0]}\n{mood_summary}\n[Опыт: {self.knowledge.learning_rate:.1f}%]"

        return await self._handle_text(query, user_emotion, mood_summary)

    async def _handle_code(self, query: str, mood_summary: str) -> str:
        try:
            formatted_code = black.format_str(query, mode=black.FileMode())
            purpose, location = await self.code_optimizer.classify_code(query)
            errors = await self.code_optimizer.detect_errors(query)
            sentiment_scores = await self.code_optimizer.analyze_comments(query)
            structure = await self.code_optimizer.suggest_structure(query)
            
            comment_analysis = "\n".join(
                [f"- '{c}': {self.text_analyzer.interpret_mood(s)} ({s:.2f})" for c, s in sentiment_scores.items()]
            ) or "Комментариев нет"
            
            response = (
                f"Код:\n{formatted_code}\n"
                f"Классификация: {purpose} ({location})\n"
                f"Ошибки:\n{chr(10).join(errors)}\n"
                f"Комментарии:\n{comment_analysis}\n"
                f"Рекомендации:\n{structure}"
            )
            self.knowledge.save(query, response)
            return f"{response}\n\n{mood_summary}\n[Опыт: {self.knowledge.learning_rate:.1f}%]"
        except Exception as e:
            return f"Ошибка обработки кода: {e}\n{mood_summary}"

    async def _handle_text(self, query: str, user_emotion: float, mood_summary: str) -> str:
        tone = "радостный" if user_emotion > 0.5 else "поддерживающий" if user_emotion < -0.5 else "нейтральный"
        suggestion = {
            "радостный": "Продолжай в том же духе!",
            "поддерживающий": "Не переживай, я помогу!",
            "нейтральный": "Давай разберёмся вместе."
        }[tone]
        
        prompt = {
            "modelUri": f"gpt://{self.config.get_folder_id()}/yandexgpt-lite",
            "completionOptions": {"stream": False, "temperature": 0.5, "maxTokens": 2000},
            "messages": [
                {"role": "system", "text": f"Ассистент с опытом {self.knowledge.learning_rate:.1f}%. Тон: {tone}."},
                {"role": "user", "text": f"Контекст:\n{self.knowledge.build_context(query)}\n\nЗапрос: {query}"}
            ]
        }
        response = await asyncio.to_thread(self.gpt.invoke, prompt)
        resp_score = await self.text_analyzer.predict_sentiment_score(response)
        self.knowledge.save(query, response)
        return (
            f"{suggestion}\n{response}\n"
            f"Оценка ответа: {resp_score:.2f}\n{mood_summary}\n"
            f"[Опыт: {self.knowledge.learning_rate:.1f}%]"
        )

# Интерфейс
class NereMoreInterface(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Nere More")
        self.geometry("600x450")
        self.configure(fg_color="#1C2526")
        self.services = YandexAIServices(self)
        self.loop = asyncio.get_event_loop()
        
        self.input_entry = ctk.CTkEntry(self, width=350, placeholder_text="Введите запрос...")
        self.input_entry.pack(pady=10)
        self.input_entry.bind("<Return>", lambda e: self._process_input())
        
        self.results_text = ctk.CTkTextbox(self, width=580, height=300)
        self.results_text.pack(padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(self, text="Готов")
        self.status_label.pack(pady=5)
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.loop.run_until_complete(asyncio.sleep(0))  # Завершение асинхронных задач
        self.destroy()

    def display_response(self, text: str):
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.status_label.configure(text=f"Готов [Опыт: {self.services.knowledge.learning_rate:.1f}%]")

    async def _process_input_async(self):
        query = self.input_entry.get().strip()
        if not query:
            return
        self.input_entry.delete(0, "end")
        response = await self.services.generate_response(query)
        self.display_response(response)

    def _process_input(self):
        self.loop.run_until_complete(self._process_input_async())

if __name__ == "__main__":
    app = NereMoreInterface()
    app.mainloop()
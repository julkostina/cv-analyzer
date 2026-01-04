# CV Analyzer API

FastAPI бекенд для аналізу резюме з використанням LangChain.

## Налаштування

1. Створіть віртуальне середовище:
   
   python -m venv .venv
   source .venv/bin/activate  # На Windows: .venv\Scripts\activate
   2. Встановіть залежності:
   
   pip install -e .
   3. Скопіюйте змінні оточення:
   
   cp .env.example .env
   # Відредагуйте .env з вашими API ключами
   4. Запустіть сервер:
   
   uvicorn app.main:app --reload
   
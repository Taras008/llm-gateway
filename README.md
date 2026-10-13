# llm-gateway

![CI](https://github.com/Taras008/llm-gateway/actions/workflows/ci.yml/badge.svg)

FastAPI-сервіс над LLM: чат, стрімінг, structured output, обробка збоїв провайдера
і облік кожного виклику (токени, латентність, помилки) у PostgreSQL.

Провайдер прихований за інтерфейсом, тож Ollama (локально, безплатно) можна замінити
на OpenAI чи Anthropic, не чіпаючи ендпоінти. Усі тести працюють без LLM і без мережі.

## Можливості

| Ендпоінт | Що робить |
|---|---|
| `GET /health` | стан сервісу й активний провайдер |
| `POST /chat` | відповідь моделі + токени й латентність |
| `POST /chat/stream` | та сама відповідь, але стрімом по шматках |
| `POST /analyze` | аналіз відгуку → JSON за Pydantic-схемою (тональність, теми, резюме) |

## Архітектура

```
HTTP ─► main.py            ендпоінти, валідація (schemas.py), LLMError → 502/503/504
          │
          ▼
        llm_calls.py       сервіс: таймер, парсинг structured output, запис логу
          │         └────► request_log.py ─► PostgreSQL (llm_request_logs)
          ▼
        llm/factory.py     вибір провайдера з конфігу
          │
          ▼
        llm/retry.py       retry з експоненційним backoff (лише для недоступності)
          │
          ▼
        llm/ollama.py      HTTP до Ollama; помилки httpx → власні LLMError
```

## Інженерні рішення

- **Провайдер за `Protocol`** + `Depends`: у тестах підміняється `FakeProvider`, у проді — Ollama.
- **Власні винятки замість httpx** (`LLMUnavailableError`, `LLMTimeoutError`, `LLMBadResponseError`):
  HTTP-шар не знає про бібліотеку клієнта; кожен тип має свій статус (503 / 504 / 502).
- **Retry лише для `LLMUnavailableError`.** Таймаут не повторюємо (клієнт уже чекав 60 с),
  невалідну відповідь теж (з `temperature=0` вона повториться). Стрім не повторюємо зовсім.
- **Вихід LLM — недовірений ввід.** Structured output завжди валідується Pydantic;
  невдача → 502, сира відповідь іде в лог (обрізана), а не клієнту.
- **Облік кожного виклику**, зокрема невдалих: невалідний JSON логується з витраченими токенами.
  Текст запитів не зберігається (персональні дані) — лише його довжина.
- **Збій запису логу не ламає відповідь** користувачу.
- **Стрімінг:** перший чанк отримується до відправки заголовків, тож недоступний провайдер
  дає 503, а не обірваний 200.

## Запуск

Потрібні Docker і [Ollama](https://ollama.com) (на Mac — нативно, бо в Docker немає доступу до GPU).

```bash
ollama pull llama3.2:3b
docker compose up --build
```

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG in one sentence"}'

curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" \
  -d '{"message": "Write a haiku about Postgres"}'
```

Документація API: http://127.0.0.1:8000/docs

### Локальна розробка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ruff
docker compose up -d --wait db
alembic upgrade head
uvicorn app.main:app --reload
```

## Тести та якість

```bash
pytest -v          # без LLM і без БД
ruff check . && ruff format --check .
alembic check      # моделі й міграції не розійшлися
```

CI (GitHub Actions) запускає лінтер, міграції на справжньому PostgreSQL і всі тести на кожен push.

## Конфігурація

| Змінна | За замовчуванням |
|---|---|
| `LLM_PROVIDER` | `ollama` (`fake` — без моделі) |
| `OLLAMA_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.2:3b` |
| `DATABASE_URL` | `postgresql+asyncpg://gateway:gateway@localhost:5433/gateway` |

## Стек

Python 3.14 · FastAPI · Pydantic · httpx · SQLAlchemy 2 (async) · asyncpg · Alembic ·
PostgreSQL · Ollama · pytest · ruff · Docker · GitHub Actions

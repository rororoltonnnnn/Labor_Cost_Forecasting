import sys, os, re, pickle, functools
from pathlib import Path

try:
    import faiss
except ImportError:
    faiss = None
# Добавляем родительскую папку в путь, чтобы импортировать из корня проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Загружаем переменные окружения из .env в корне проекта
load_dotenv(Path(__file__).parent.parent / ".env")

# Создаем приложение FastAPI и настраиваем шаблоны
app = FastAPI()
jinja_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")), cache_size=0)
templates = Jinja2Templates(env=jinja_env)

# Название модели для создания эмбеддингов
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Как будем сохранять кешированный FAISS индекс
CACHE_FILE = "ticket_cache_faiss.pkl"

# Пороги и лимиты для поиска
SETTINGS = {
    "cosine_threshold": 0.60,      # Минимальное сходство для попадания в результаты
    "overlap_threshold": 0.20,     # Минимальное пересечение ключевых слов
    "top_k_search": 20,            # Сколько берем из FAISS за один поиск
    "top_k_final": 5,              # Сколько вернуть в финальном ответе
    "min_candidates_for_fallback": 3  # Если найдено меньше - используем AI
}

# Загружаем модель один раз и кешируем (lru_cache)
@functools.lru_cache(maxsize=1)
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        dim = model.get_embedding_dimension()
        print(f"[INFO] Model loaded. Embedding dimension: {dim}")
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None


# Получаем или создаем FAISS индекс и кешируем его
@functools.lru_cache(maxsize=1)
def get_faiss():
    # Попытка импорта FAISS и pickle
    try:
        import faiss
        import pickle
    except ImportError:
        return None, None, None, None, None

    model = load_model()
    if model is None:
        return None, None, None, None, None

    model_dim = model.get_embedding_dimension()
    print(f"[INFO] Model dimension: {model_dim}")

    cache_path = Path(__file__).parent / CACHE_FILE

    # Если есть кеш - пробуем загрузить
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            idx = cache["index"]
            print(f"[INFO] Cache found. Index dimension: {idx.d}")
            # Проверяем совместимость кеша с текущей моделью
            if idx.d == model_dim and "texts" in cache and len(cache["subjects"]) == len(cache["texts"]):
                print("[INFO] Using cached FAISS index")
                return idx, cache["hours"], cache["subjects"], cache["descriptions"], cache["texts"]
            else:
                print(f"[WARN] Incompatible cache (dimension, missing texts, or size mismatch). Rebuilding...")
        except Exception as e:
            print(f"[WARN] Cache corrupted: {e}. Rebuilding...")

    # Строим индекс заново из CSV
    print("[INFO] Building new FAISS index from scratch...")
    df = pd.read_csv("../customer_support_tickets.csv")
    df = df[df["Ticket Status"] == "Closed"].copy()

    prod_col = "Product Purchased"
    for col in ["Ticket Subject", "Ticket Description"]:
        df[col] = df.apply(
            lambda r: re.sub(
                r"\{.*?[Pp]roduct.*?[Pp]urchased.*?\}",
                str(r[prod_col]) if pd.notna(r[prod_col]) else "Product",
                str(r[col])
            ),
            axis=1
        )

    def clean_text(text):
        # Очищаем строку: обрезаем пробелы, убираем лишние, переводим в нижний регистр
        if not isinstance(text, str):
            return ""
        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("{product_purchased}", "product")
        return text.lower()

    df["text"] = df[["Ticket Subject", "Ticket Description", "Ticket Type"]].fillna("").agg(" ".join, axis=1).apply(clean_text)

    def get_h(row):
        # Вычисляем время решения в часах из разницы дат
        try:
            r = pd.to_datetime(row["Time to Resolution"])
            if pd.notna(row.get("First Response Time")):
                f = pd.to_datetime(row["First Response Time"])
                h = (r - f).total_seconds() / 3600
                if 0 < h < 100:
                    return h
        except Exception:
            pass
        return 0

    df["hours"] = df.apply(get_h, axis=1)
    df = df[(df["hours"] > 0) & (df["hours"] < 100)].copy()
    df = df[df["text"].str.len() > 10].copy()

    subjects = df["Ticket Subject"].tolist()
    descriptions = df["Ticket Description"].tolist()
    texts = df["text"].tolist()
    hours = df["hours"].tolist()

    print(f"[INFO] Encoding {len(texts)} texts with {MODEL_NAME}...")
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64).astype("float32")
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # Нормализуем эмбеддинги и создаем FAISS индекс
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"[INFO] Index built. Dim: {index.d}, Total vectors: {index.ntotal}")

    cache = {
        "index": index,
        "hours": hours,
        "subjects": subjects,
        "descriptions": descriptions,
        "texts": texts
    }
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[INFO] Cache saved to {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cache: {e}")

    return index, hours, subjects, descriptions, texts


def ask_ai(text):
    # Запрашиваем у внешней LLM оценку времени решения
    try:
        from openai import OpenAI
        client = OpenAI(base_url='https://polza.ai/api/v1', api_key=os.getenv('POLZA_API_KEY'))
        response = client.chat.completions.create(
            model="qwen/qwen3.6-plus",
            messages=[{"role": "user", "content": f"You are a senior support engineer. Estimate resolution time in hours. Return only a number. Issue: {text}"}],
            max_tokens=10
        )
        numbers = re.findall(r'\d+\.?\d*', response.choices[0].message.content)
        if numbers:
            val = float(numbers[0])
            if 0 <= val <= 100:
                return val
    except Exception as e:
        print(f"[ERROR] AI request failed: {e}")
    return None


def clean_text_v2(text: str) -> str:
    # Убираем лишнее из текста: пунктуацию, пробелы, приводим к нижнему регистру
    if not isinstance(text, str):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def keyword_overlap(query: str, candidate: str) -> float:
    # Считаем, какую часть слов запроса есть в тексте кандидата
    q_words = set(clean_text_v2(query).split())
    c_words = set(clean_text_v2(candidate).split())
    if not q_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


def get_expected_type(query: str) -> str:
    """Определяем тип обращения по ключевым словам: technical, billing, product или cancellation."""
    q = clean_text_v2(query)
    tech = {"technical", "issue", "error", "broken", "not working", "fix", "repair", "setup", "install", "printer", "computer", "server", "software"}
    billing = {"billing", "payment", "invoice", "charge", "refund", "money", "pay", "card"}
    product = {"product", "purchase", "buy", "order", "delivery", "ship", "item"}
    cancel = {"cancel", "terminate", "subscription", "membership", "return"}
    if any(w in q for w in tech):
        return "technical"
    if any(w in q for w in billing):
        return "billing"
    if any(w in q for w in product):
        return "product"
    if any(w in q for w in cancel):
        return "cancellation"
    return "unknown"


def do_search_v2(query: str):
    """
    Поиск похожих тикетов:
    1. Берем топ-20 из FAISS
    2. Фильтруем по сходству и пересечению слов
    3. Переранжируем по комбинированному скору
    4. Если найдено мало - используем AI
    """
    res = get_faiss()
    if res[0] is None:
        print("[WARN] FAISS index not available")
        return [], [], [], [], [], [], 0

    idx = res[0]
    hrs = res[1]
    subj = res[2]
    desc = res[3]
    texts = res[4] if len(res) > 4 else [""] * len(hrs)

    model = load_model()
    if model is None:
        print("[WARN] Model not available")
        return [], [], [], [], [], [], 0

    query_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    k_search = SETTINGS["top_k_search"]
    k_actual = min(k_search, len(hrs))
    distances, indices = idx.search(query_vec, k_actual)

    similarities = distances[0].tolist()
    candidates = []

    query_clean = clean_text_v2(query)
    expected_type = get_expected_type(query)

    for idx_pos, sim in zip(indices[0], similarities):
        # Пропускаем если сходство ниже порога
        if sim < SETTINGS["cosine_threshold"]:
            continue

        ticket_text = texts[idx_pos] if texts else ""
        overlap = keyword_overlap(query_clean, ticket_text)

        # Пропускаем если пересечение слов мало
        if overlap < SETTINGS["overlap_threshold"]:
            continue

        type_match = 1 if expected_type in ticket_text else 0

        score = 0.6 * sim + 0.25 * overlap + 0.15 * type_match

        candidates.append({
            "idx": idx_pos,
            "similarity": sim,
            "overlap": overlap,
            "type_match": type_match,
            "score": score,
            "hours": hrs[idx_pos],
            "subject": subj[idx_pos],
            "description": desc[idx_pos]
        })

    count_total = len(candidates)

    # Если найдено мало подходящих - переключаемся на AI
    if count_total < SETTINGS["min_candidates_for_fallback"]:
        print(f"[INFO] Only {count_total} candidates found (< {SETTINGS['min_candidates_for_fallback']}). AI fallback.")
        return [], [], [], [], [], [], 0

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_k = min(SETTINGS["top_k_final"], len(candidates))
    final = candidates[:top_k]

    hours_list = [c["hours"] for c in final]
    similarities_list = [c["similarity"] for c in final]
    subjects = [c["subject"] for c in final]
    descriptions = [c["description"] for c in final]
    scores = [c["score"] for c in final]
    overlaps = [c["overlap"] for c in final]

    confidence = float(np.mean(similarities_list)) if similarities_list else 0.0

    print(f"[INFO] Found {len(final)} tickets after re-ranking. Confidence: {confidence:.3f}")
    return hours_list, similarities_list, subjects, descriptions, scores, overlaps, len(final)


# Главная страница: ищем похожие тикеты и выводим прогноз времени решения
@app.get("/")
async def home(request: Request, q: str = None, use_ai: bool = False):
    result = None
    error = None
    if q:
        word_count = len(q.strip().split())
        # Защита от коротких запросов
        if word_count <= 3:
            error = "Запрос слишком короткий. Опишите проблему минимум 4 слова, чтобы получить точный прогноз."
        else:
            hours_list, similarities, subjects, descriptions, scores, overlaps, count = do_search_v2(q)
            if count >= SETTINGS["min_candidates_for_fallback"]:
                total_weight = sum(scores)
                if total_weight > 0:
                    weighted_sum = sum(h * s for h, s in zip(hours_list, scores))
                    weighted_avg = weighted_sum / total_weight
                else:
                    weighted_avg = float(np.median(hours_list))
                # Ограничиваем ответ в разумные пределы
                final_est = max(0.5, min(weighted_avg, 48))
                confidence = float(np.mean(similarities)) if similarities else 0.0
                ai_h = None
                if use_ai:
                    ai_h = ask_ai(q)
                    if ai_h:
                        final_est = (final_est + ai_h) / 2
                # Формируем результат со всеми найденными тикетами
                result = {
                    "final": final_est,
                    "confidence": confidence,
                    "count": count,
                    "tickets": [
                        {"s": s, "d": d, "h": h, "sc": sc, "scr": scr, "ol": ol}
                        for s, d, h, sc, scr, ol in zip(subjects, descriptions, hours_list, similarities, scores, overlaps)
                    ],
                    "ai": ai_h
                }
            else:
                # Мало кандидатов - полагаемся на AI полностью
                ai_h = ask_ai(q)
                if ai_h and 0 <= ai_h <= 100:
                    result = {
                        "final": max(0.5, min(ai_h, 48)),
                        "confidence": 0.0,
                        "count": 0,
                        "tickets": [],
                        "ai": ai_h,
                        "ai_fallback": True
                    }
                else:
                    result = {
                        "final": 5.0,
                        "confidence": 0.0,
                        "count": 0,
                        "tickets": [],
                        "ai": None,
                        "ai_fallback": True
                    }
    return templates.TemplateResponse(request, "index.html", context={"request": request, "result": result, "q": q or "", "use_ai": use_ai, "error": error})


@app.get("/dataset")
# Страница с данными: выводим тикеты постранично с поиском
async def dataset(request: Request, page: int = 1, per_page: int = 20, q: str = None):
    df = pd.read_csv("../customer_support_tickets.csv")

    # То же заменяем шаблоны на реальные значения
    prod_col = "Product Purchased"
    for col in ["Ticket Subject", "Ticket Description"]:
        df[col] = df.apply(
            lambda r: re.sub(
                r"\{.*?[Pp]roduct.*?[Pp]urchased.*?\}",
                str(r[prod_col]) if pd.notna(r[prod_col]) else "Product",
                str(r[col])
            ),
            axis=1
        )

    def get_h(row):
        # Считаем часы до решения тикета
        try:
            r = pd.to_datetime(row["Time to Resolution"])
            if pd.notna(row.get("First Response Time")):
                f = pd.to_datetime(row["First Response Time"])
                h = (r - f).total_seconds() / 3600
                if 0 < h < 100:
                    return round(h, 1)
        except Exception:
            pass
        return None

    df["hours"] = df.apply(get_h, axis=1)
    df = df[df["hours"].notna()].copy()

    # Фильтр по поисковому запросу
    if q:
        mask_subject = df["Ticket Subject"].fillna("").str.contains(q, case=False, na=False)
        mask_desc = df["Ticket Description"].fillna("").str.contains(q, case=False, na=False)
        df = df[mask_subject | mask_desc]

    total = len(df)
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    tickets = df.iloc[start:end].to_dict("records")

    return templates.TemplateResponse(request, "dataset.html", context={
        "request": request,
        "tickets": tickets,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total": total,
        "search_query": q
    })

# Запуск: uvicorn main:app --host 127.0.0.1 --port 8000
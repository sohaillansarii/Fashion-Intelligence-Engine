# main.py
import os
import io
import re
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download
import faiss
import pandas as pd

# ──────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────
HF_REPO = "sohaillansarii/fashion-search"          # <-- change this
MODEL_ID  = "openai/clip-vit-base-patch32"        # base CLIP (fine-tuned later if needed)
TOP_K     = 5
HF_IMAGE_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/images"   # <-- ADD THIS LINE
app = FastAPI(title="Fashion Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# 2. GLOBAL STATE (loaded once at startup)
# ──────────────────────────────────────────────
state = {
    "model": None,
    "processor": None,
    "embeddings": None,
    "valid_ids": None,
    "df": None,
    "index": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def download_from_hf(filename: str, subfolder: str | None = None) -> str:
    """Download a file from Hugging Face Hub to a local cache path."""
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")


@app.on_event("startup")
def load_resources():
    """Download model + embeddings from HF Hub and load into memory."""
    print("⏳ Downloading resources from Hugging Face...")

    # --- CLIP model & processor ---
    state["model"] = CLIPModel.from_pretrained(MODEL_ID).to(state["device"])
    state["processor"] = CLIPProcessor.from_pretrained(MODEL_ID)
    state["model"].eval()

    # --- Embeddings + metadata from HF Hub ---
    emb_path = download_from_hf("image_embeddings_optuna.npy")
    ids_path = download_from_hf("valid_ids_optuna.npy")
    csv_path = download_from_hf("cleaned_styles.csv")

    state["embeddings"] = np.load(emb_path).astype("float32")
    state["valid_ids"]  = np.load(ids_path, allow_pickle=True)
    state["df"]         = pd.read_csv(csv_path)

    # --- Build FAISS index ---
    dim = state["embeddings"].shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(state["embeddings"])
    state["index"] = index

    print(f"✅ Loaded {len(state['valid_ids'])} products, FAISS index ready.")


# ──────────────────────────────────────────────
# 3. HELPERS (ported from your notebook)
# ──────────────────────────────────────────────
def get_text_features(text: str):
    inputs = state["processor"](text=[text], return_tensors="pt", padding=True).to(state["device"])
    with torch.no_grad():
        feats = state["model"].get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def get_image_features(image: Image.Image):
    inputs = state["processor"](images=image, return_tensors="pt").to(state["device"])
    with torch.no_grad():
        feats = state["model"].get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def faiss_search(query_vector: np.ndarray, k: int = TOP_K):
    scores, indices = state["index"].search(query_vector, k)
    return scores[0], indices[0]


# ── Query parser (from your notebook) ──
def parse_query(query: str, df: pd.DataFrame):
    q = query.lower()
    filters = {}

    price_match = re.search(r'(?:under|below|less than)\s*\$?(\d+)', q)
    if price_match:
        filters['max_price'] = int(price_match.group(1))

    known_colors = df['baseColour'].dropna().str.lower().unique().tolist()
    for color in known_colors:
        if color in q:
            filters['color'] = color
            break

    known_cats = df['articleType'].dropna().str.lower().unique().tolist()
    for cat in known_cats:
        cat_singular = cat.rstrip('s')
        if cat in q or cat_singular in q:
            filters['category'] = cat
            break

    if 'women' in q or "women's" in q:
        filters['gender'] = 'Women'
    elif 'men' in q or "men's" in q:
        filters['gender'] = 'Men'

    return filters


def apply_filters(df: pd.DataFrame, filters: dict):
    filtered = df.copy()
    if 'max_price' in filters:
        filtered = filtered[filtered['price'] <= filters['max_price']]
    if 'color' in filters:
        filtered = filtered[filtered['baseColour'].str.lower() == filters['color']]
    if 'category' in filters:
        filtered = filtered[filtered['articleType'].str.lower() == filters['category']]
    if 'gender' in filters:
        filtered = filtered[filtered['gender'] == filters['gender']]
    return filtered


def format_row(row) -> dict:
    return {
        "id": int(row["id"]),
        "productDisplayName": str(row.get("productDisplayName", "")),
        "articleType": str(row.get("articleType", "")),
        "baseColour": str(row.get("baseColour", "")),
        "gender": str(row.get("gender", "")),
        "price": float(row.get("price", 0)),
        "usage": str(row.get("usage", "")),
        "season": str(row.get("season", "")),
    }


# ──────────────────────────────────────────────
# 4. ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Fashion Search API",
        "endpoints": {
            "/search?query=...": "Text-based search",
            "/refine": "Image + text refinement (POST, multipart)",
            "/health": "Health check",
        },
    }


@app.get("/health")
def health():
    ready = all(v is not None for v in [
        state["model"], state["embeddings"], state["index"], state["df"]
    ])
    return {"status": "ready" if ready else "loading", "device": state["device"]}


@app.get("/search")
def search(query: str, top_k: int = TOP_K):
    """Text search with automatic filter parsing."""
    if not query.strip():
        raise HTTPException(400, "query is required")

    filters = parse_query(query, state["df"])
    filtered_df = apply_filters(state["df"], filters)

    # If filters are too strict, fall back to full catalog
    if len(filtered_df) == 0:
        filtered_df = state["df"]

    allowed_ids = set(filtered_df["id"])
    id_to_idx = {pid: i for i, pid in enumerate(state["valid_ids"]) if pid in allowed_ids}

    if not id_to_idx:
        return {"query": query, "filters": filters, "results": []}

    # Build a mini FAISS index over allowed items
    allowed_indices = list(id_to_idx.values())
    mini_embeddings = state["embeddings"][allowed_indices].copy()
    mini_index = faiss.IndexFlatIP(mini_embeddings.shape[1])
    mini_index.add(mini_embeddings)

    query_vec = get_text_features(query)
    scores, indices = mini_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = filtered_df[filtered_df["id"] == state["valid_ids"][allowed_indices[idx]]].iloc[0]
        results.append({**format_row(row), "score": float(score)})

    return {"query": query, "filters": filters, "results": results}


@app.post("/refine")
async def refine(
    image: UploadFile = File(...),
    text: str = Form(""),
    image_weight: float = Form(0.5),
    top_k: int = Form(TOP_K),
):
    """Refine search using an uploaded image + optional text."""
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    img_feats = get_image_features(img)
    if text.strip():
        txt_feats = get_text_features(text)
        combined = image_weight * img_feats + (1 - image_weight) * txt_feats
    else:
        combined = img_feats

    combined = combined / np.linalg.norm(combined, axis=-1, keepdims=True)

    scores, indices = faiss_search(combined, top_k)
    results = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        row = state["df"][state["df"]["id"] == state["valid_ids"][idx]].iloc[0]
        results.append({**format_row(row), "score": float(score)})

    return {"text": text, "image_weight": image_weight, "results": results}


def format_row(row) -> dict:
    return {
        "id": int(row["id"]),
        "productDisplayName": str(row.get("productDisplayName", "")),
        "articleType": str(row.get("articleType", "")),
        "baseColour": str(row.get("baseColour", "")),
        "gender": str(row.get("gender", "")),
        "price": float(row.get("price", 0)),
        "usage": str(row.get("usage", "")),
        "season": str(row.get("season", "")),
       "image_url": get_image_url(int(row['id'])),  # <-- ADD THIS LINE
    }
    
def get_image_url(product_id: int) -> str:
    bucket = product_id // 5000
    return f"{HF_IMAGE_BASE}/bucket_{bucket}/{product_id}.jpg"
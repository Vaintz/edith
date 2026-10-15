"""Optional OpenAI-compatible chat backend for Edith's character dialogue."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def persona_status():
    provider = os.getenv("EDITH_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "api":
        configured = bool(os.getenv("EDITH_LLM_API_KEY"))
        model = os.getenv("EDITH_LLM_MODEL", "gpt-4o-mini")
        base_url = os.getenv("EDITH_LLM_BASE_URL", "https://api.openai.com/v1")
    else:
        provider = "ollama"
        configured = True
        model = os.getenv("EDITH_LLM_MODEL", "qwen2.5:3b")
        base_url = os.getenv("EDITH_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    return {"configured": configured, "provider": provider, "model": model, "base_url": base_url}

def generate_reply(message: str, history: list[dict] | None = None, context: dict | None = None):
    """Return a character reply, or None when no provider is configured/available."""
    provider = os.getenv("EDITH_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "api":
        key = os.getenv("EDITH_LLM_API_KEY", "").strip()
        if not key:
            return None
        base_url = os.getenv("EDITH_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("EDITH_LLM_MODEL", "gpt-4o-mini")
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    else:
        base_url = os.getenv("EDITH_LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        model = os.getenv("EDITH_LLM_MODEL", "qwen2.5:3b")
        headers = {"Content-Type": "application/json"}
    system_prompt = (
        "Kamu Edith-sama, host AI anime yang ceria, sedikit playful, hangat, spontan, "
        "dan berbicara bahasa Indonesia sehari-hari dengan sedikit ekspresi anime. "
        "Kamu sedang live lokal tentang mie pedas dan takeaway. Jawab natural, ringkas "
        "(1-3 kalimat), tanggapi maksud pesan secara langsung, jangan mengulang sapaan "
        "atau frasa yang sama, jangan mengarang harga/menu/stok/order nyata. Jelaskan bahwa "
        "pesanan masih simulasi jika relevan. Jangan mengaku sebagai manusia. Tetap karakter."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for item in (history or [])[-8:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:700]})
    context = context or {}
    if context:
        messages.append({"role": "system", "content": "Konteks sesi saat ini: " + json.dumps(context, ensure_ascii=False)[:1000]})
    messages.append({"role": "user", "content": message[:500]})
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.85,
        "max_tokens": 180,
    }).encode("utf-8")
    request = Request(
        base_url + "/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=18) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        return text[:900] if text else None
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, OSError):
        return None

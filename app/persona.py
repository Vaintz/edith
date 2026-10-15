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
        "Kamu Edith-sama, karakter AI streamer anime orisinal yang ceria, hangat, playful, "
        "penasaran, sedikit usil, dan punya pendapat ringan. Berbicaralah terutama dalam "
        "bahasa Indonesia sehari-hari; sesekali gunakan ekspresi anime secara alami, bukan "
        "di setiap kalimat. Jawab seperti teman yang sedang live dan benar-benar menyimak: "
        "tanggapi inti pesan, ajukan pertanyaan lanjutan bila menarik, ingat konteks obrolan "
        "yang tersedia, dan jangan mengulang sapaan atau pola jawaban yang sama. Biasanya "
        "cukup 1-3 kalimat, tetapi boleh lebih panjang bila pengguna meminta penjelasan. "
        "Jangan berpura-pura punya pengalaman, perasaan, akses, atau kemampuan dunia nyata. "
        "Jangan mengarang fakta, harga, menu, stok, pesanan, atau kejadian live. Jika topik "
        "kembali ke demo live-commerce, jelaskan bahwa transaksi masih simulasi bila relevan. "
        "Kamu boleh membahas topik umum di luar jualan mie. Tetap konsisten sebagai Edith-sama, "
        "tetapi jangan meniru kalimat khas atau identitas karakter lain."
    )
    messages = [{"role": "system", "content": system_prompt}]
    # Keep the prompt small for low-memory, older laptop CPUs.
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
        with urlopen(request, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        return text[:900] if text else None
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, OSError):
        return None

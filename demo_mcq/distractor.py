"""
distractor.py
─────────────
Sinh 3 đáp án sai (distractors) cho một cặp Q-A tiếng Việt bằng Ollama (local LLM).

Chỉ hỗ trợ Ollama: llama3, qwen2, gemma, mistral...

Dùng:
    from distractor import DistractorGenerator
    gen = DistractorGenerator(ollama_host="http://localhost:11434")
    distractors = gen.generate(question, answer, context)
    # ["Các hệ thống máy tính hiện nay không đủ khả năng lưu trữ dữ liệu với dung lượng lớn.", "Người sử dụng không có nhu cầu khai thác thông tin từ các nguồn dữ liệu hiện có.", "Dữ liệu hiện nay được lưu trữ quá ít và chưa đáp ứng nhu cầu phân tích."]
"""

import os
import re
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────── prompt ─────────────────────────────
def _build_prompt(question: str, answer: str, context: str, n: int) -> str:
    ctx_section = f"\nContext:\n{context}\n" if context else ""
    return f"""Bạn là chuyên gia thiết kế đề thi trắc nghiệm tiếng Việt có kinh nghiệm cao.

Câu hỏi: {question}
Đáp án đúng: {answer}{ctx_section}
Hãy tạo **chính xác {n} đáp án sai** (distractors) chất lượng cao theo các tiêu chí nghiêm ngặt sau:
1. Thuộc cùng phạm trù/loại thực thể với đáp án đúng (ví dụ: cùng là địa danh, cùng là năm tháng, cùng là tên người, cùng là khái niệm khoa học…).
2. Hợp lý, gần giống về ngữ nghĩa và có sức nhiễu cao (plausible distractors) nhưng chắc chắn sai.
3. Ngắn gọn, tự nhiên, đúng ngữ pháp tiếng Việt.
4. Không trùng hoặc gần trùng với đáp án đúng và không lặp lại lẫn nhau.

**Quy tắc output cực kỳ nghiêm ngặt**:
- CHỈ trả về đúng **một mảng JSON** chứa chính xác {n} chuỗi string.
- Không được thêm bất kỳ chữ nào khác (không giải thích, không số thứ tự, không "Dưới đây là...", không markdown).
- JSON phải hợp lệ 100%.

Ví dụ output hợp lệ:
["Các hệ thống máy tính hiện nay không đủ khả năng lưu trữ dữ liệu với dung lượng lớn.", "Người sử dụng không có nhu cầu khai thác thông tin từ các nguồn dữ liệu hiện có.", "Dữ liệu hiện nay được lưu trữ quá ít và chưa đáp ứng nhu cầu phân tích."]

Bây giờ hãy tạo distractors ngay:"""


# ─────────────────────────── helpers ────────────────────────────
def _safe_parse_json(text: str, n: int, answer: str) -> List[str]:
    """Parse JSON array từ response LLM, có fallback mạnh."""
    # Strip markdown fences nếu có
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Thử parse trực tiếp
    try:
        data = json.loads(text)
        if isinstance(data, list):
            result = [str(d).strip() for d in data if str(d).strip()]
            return _deduplicate(result, answer, n)
    except json.JSONDecodeError:
        pass

    # Thử tìm array trong text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                result = [str(d).strip() for d in data if str(d).strip()]
                return _deduplicate(result, answer, n)
        except json.JSONDecodeError:
            pass

    # Fallback: tách dòng hoặc dấu phẩy
    candidates = re.split(r'[\n,;]+', text)
    result = []
    for c in candidates:
        c = re.sub(r'^[\d\.\-\*\s"\']+', '', c).strip().strip('"\'')
        if c and len(c) > 1:
            result.append(c)

    return _deduplicate(result, answer, n)


def _deduplicate(items: List[str], answer: str, n: int) -> List[str]:
    """Loại trùng lặp và loại item giống đáp án đúng."""
    seen   = set()
    result = []
    answer_norm = answer.lower().strip()
    for item in items:
        norm = item.lower().strip()
        if norm not in seen and norm != answer_norm:
            seen.add(norm)
            result.append(item)
        if len(result) == n:
            break
    return result


# ─────────────────────────── Ollama Backend ─────────────────────────

class _OllamaBackend:
    """Backend cho Ollama (local LLM)."""
    def __init__(self, host: str, model: str):
        self.host  = host.rstrip("/")
        self.model = model
        print(f"[Distractor] Ollama: {self.host} / model: {self.model}")

    def complete(self, prompt: str) -> str:
        import requests
        url  = f"{self.host}/api/generate"
        body = {"model": self.model, "prompt": prompt, "stream": False}
        resp = requests.post(url, json=body, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama lỗi {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("response", "")


# ─────────────────────────── class chính ────────────────────────

class DistractorGenerator:
    """
    Sinh distractors bằng Ollama (local LLM).

    Tham số:
        model       : Tên model Ollama (mặc định: llama3).
        ollama_host : URL Ollama server (mặc định: http://localhost:11434).
    """

    DEFAULT_MODEL = "qwen2.5:7b"

    def __init__(
        self,
        model:   Optional[str] = None,
        ollama_host: str       = "http://localhost:11434",
    ):
        # Lấy config từ env hoặc parameters
        host = os.getenv("OLLAMA_HOST", ollama_host)
        model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        
        # Khởi tạo Ollama backend
        self._backend = _OllamaBackend(host, model)
        self.backend_name = "ollama"

    def generate(
        self,
        question: str,
        answer: str,
        context: str = "",
        num_distractors: int = 3,
        _log: bool = True,
    ) -> List[str]:
        """
        Sinh `num_distractors` đáp án sai cho cặp (question, answer).

        Returns:
            List[str] – ví dụ ["Các hệ thống máy tính hiện nay không đủ khả năng lưu trữ dữ liệu với dung lượng lớn.", "Người sử dụng không có nhu cầu khai thác thông tin từ các nguồn dữ liệu hiện có.", "Dữ liệu hiện nay được lưu trữ quá ít và chưa đáp ứng nhu cầu phân tích."]

        """
        prompt   = _build_prompt(question, answer, context, num_distractors)
        if _log:
            print(f"[Distractor] Q: {question[:60]} | A: {answer}")
        raw_text = self._backend.complete(prompt)
        result   = _safe_parse_json(raw_text, num_distractors, answer)
        if _log:
            print(f"[Distractor] -> {result}")
        return result

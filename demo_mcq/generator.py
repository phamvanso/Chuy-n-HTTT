"""
generator.py
────────────
Sinh cặp Question–Answer từ đoạn văn tiếng Việt.

Hỗ trợ hai loại model:
  1. Multitask (QG + AE cùng 1 model)  – VD: shnl/vit5-vinewsqa-qg-ae
     • Bước 1 (AE): "extract answers: <context>"   → đáp án, cách nhau bằng '<sep>'
     • Bước 2 (QG): "generate question: <ctx> <hl> <answer> <hl>"  → câu hỏi

  2. Pipeline (QG-only model) – VD: namngo/pipeline-vit5-viquad-qg
     • Bước 1 (AE): tách câu văn làm đáp án ứng viên
     • Bước 2 (QG): "generate question: <ctx> <hl> <answer> <hl>"  → câu hỏi

Dùng:
    from generator import QAGenerator
    gen = QAGenerator()
    pairs = gen.generate("Hà Nội là thủ đô…")
    # [{"question": "…", "answer": "…"}, …]
"""

import os
import re
import unicodedata
import logging
import time
from typing import List, Dict, Optional

logging.basicConfig(level=logging.WARNING)

# ───────────────────────── hằng số ──────────────────────────────
# Model mặc định: multitask (QG + AE), còn tồn tại trên HF
DEFAULT_MODEL = "shnl/vit5-vinewsqa-qg-ae"
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 128
HL_TOKEN = "<hl>"


# ───────────────────────── helpers ──────────────────────────────
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _nfc(text: str) -> str:
    """Normalize Unicode NFC – fix lỗi so sánh chuỗi tiếng Việt."""
    return unicodedata.normalize("NFC", text)


def _answer_in_context(answer: str, context: str) -> bool:
    """Kiểm tra answer có trong context, chấp nhận sai lệch NFC."""
    a = _nfc(answer).lower()
    c = _nfc(context).lower()
    if a in c:
        return True
    # Kiểm tra word overlap >= 80% (chịu được model output normalize khác)
    words = [w for w in a.split() if len(w) >= 2]
    if not words:
        return False
    return sum(1 for w in words if w in c) / len(words) >= 0.8
def _split_sentences(text: str) -> List[str]:
    """Tách câu đơn giản theo dấu câu tiếng Việt."""
    parts = re.split(r"(?<=[.?!;])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _is_multitask(model_name: str) -> bool:
    """True nếu model có cả QG lẫn AE (tên chứa cả 'qg' và 'ae')."""
    parts = re.split(r"[-/]", model_name.lower())
    return "qg" in parts and "ae" in parts


# ───────────────────────── class chính ──────────────────────────
class QAGenerator:
    """
    Sinh Question-Answer từ context tiếng Việt dùng ViT5.

    Tham số:
        model_name : Tên model HF hub.
        use_api    : Giữ để tương thích với app.py (bị bỏ qua – luôn dùng local).
        hf_token   : HF token (cần nếu model private).
        device     : "cpu" | "cuda" | "auto".
    """

    def __init__(
        self,
        model_name: str = None,
        use_api: bool = None,
        hf_token: str = None,
        device: str = "auto",
    ):
        self.model_name = model_name or os.getenv("VIQAG_MODEL", DEFAULT_MODEL)
        self.hf_token   = hf_token   or os.getenv("HF_TOKEN", "")
        self.device     = device
        self.multitask  = _is_multitask(self.model_name)

        self._model      = None
        self._tokenizer  = None
        self._device_str = "cpu"

        self._load_local()

    # ── tải model ───────────────────────────────────────────────
    def _load_local(self):
        try:
            import torch
            from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
        except ImportError:
            raise RuntimeError(
                "Thiếu thư viện. Chạy:  pip install torch transformers sentencepiece"
            )

        token = self.hf_token or None
        print(f"[Generator] Đang load model '{self.model_name}' (lần đầu ~1–3 phút)…")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=token,
                use_fast=False,
                legacy=False,
            )
        except Exception:
            # Fallback cho một số model ViT5/T5 khi AutoTokenizer không tương thích phiên bản mới.
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                token=token,
                legacy=False,
            )

        # Thêm special token <hl> nếu chưa có
        if HL_TOKEN not in self._tokenizer.get_vocab():
            self._tokenizer.add_special_tokens({"additional_special_tokens": [HL_TOKEN]})

        try:
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, token=token
            )
        except Exception:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, token=token
            )

        self._model.resize_token_embeddings(len(self._tokenizer))

        import torch
        if self.device == "auto":
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device_str = self.device

        self._model.to(self._device_str)
        self._model.eval()
        mode = "multitask QG+AE" if self.multitask else "pipeline QG-only"
        print(f"[Generator] Model sẵn sàng trên '{self._device_str}' ({mode}).")

    # ── inference helper ─────────────────────────────────────────
    def _infer(self, prompt: str, max_new_tokens: int = MAX_OUTPUT_LEN,
               num_return_sequences: int = 1) -> List[str]:
        """Trả về list string (>=1 kết quả khi num_return_sequences>1)."""
        import torch
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(self._device_str)
        num_beams = max(4, num_return_sequences)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
            )
        return [self._tokenizer.decode(i, skip_special_tokens=True) for i in ids]

    def _infer_one(self, prompt: str, max_new_tokens: int = MAX_OUTPUT_LEN) -> str:
        return self._infer(prompt, max_new_tokens)[0]

    # ── AE: dùng model multitask ─────────────────────────────────
    def _extract_answers_multitask(self, context: str, need: int) -> List[str]:
        """
        AE: highlight từng câu → model trả nhiều candidate answers.
        Dùng num_return_sequences để lấy >=1 answer/câu khi cần thiết.
        """
        sentences = _split_sentences(context)
        answers: List[str] = []
        seen: set = set()

        # Tính số beam cần thiết để đủ `need` answers từ len(sentences) câu
        seqs_per_sent = max(1, -(-need // max(len(sentences), 1)))  # ceil division
        seqs_per_sent = min(seqs_per_sent, 4)  # tối đa 4 beam

        for sentence in sentences:
            if len(answers) >= need * 2:  # đủ candidate rồi, dừng
                break
            pos = context.find(sentence)
            if pos == -1:
                continue
            highlighted = (
                context[:pos]
                + f"{HL_TOKEN} {sentence} {HL_TOKEN}"
                + context[pos + len(sentence):]
            )
            prompt = f"extract answers: {highlighted}"
            raws = self._infer(prompt, max_new_tokens=128, num_return_sequences=seqs_per_sent)
            for raw in (_clean(r) for r in raws):
                norm = _nfc(raw).lower()
                if raw and len(raw) >= 2 and norm not in seen and _answer_in_context(raw, context):
                    seen.add(norm)
                    answers.append(raw)
                    print(f"[AE] + '{raw}'")
                elif raw:
                    print(f"[AE] drop '{raw}' (not in ctx or dup)")
        return answers

    # ── AE fallback: tách câu ────────────────────────────────────
    def _extract_answers_sentences(self, context: str, num: int) -> List[str]:
        """Fallback: dùng mỗi câu làm đáp án ứng viên."""
        return _split_sentences(context)[:num]

    # ── QG ───────────────────────────────────────────────────────
    def _generate_question(self, context: str, answer: str) -> Optional[str]:
        """
        Sinh câu hỏi cho cặp (context, answer).
        Format: "generate question: <context_với_<hl>_quanh_answer>"
        """
        pos = _nfc(context).lower().find(_nfc(answer).lower())
        if pos == -1:
            highlighted = f"{context} {HL_TOKEN} {answer} {HL_TOKEN}"
        else:
            highlighted = (
                context[:pos]
                + f"{HL_TOKEN} {context[pos:pos+len(answer)]} {HL_TOKEN}"
                + context[pos + len(answer):]
            )
        prompt = f"generate question: {highlighted}"
        q = _clean(self._infer_one(prompt))
        return q if q else None

    # ── public API ─────────────────────────────────────────────
    def generate(
        self,
        context: str,
        num_pairs: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Sinh tối đa `num_pairs` cặp Q-A từ `context`.

        Returns:
            [{"question": "…", "answer": "…"}, …]
        """
        context = _clean(context)
        if not context:
            return []

        # ── Bước 1: trích xuất đáp án ──────────────────────
        answers: List[str] = []
        if self.multitask:
            print(f"[AE] Trích xuất đáp án từ {len(context)} ky tự ({len(_split_sentences(context))} câu)...")
            answers = self._extract_answers_multitask(context, need=num_pairs * 2)
            print(f"[AE] -> {len(answers)} đáp án: {answers}")

        # Fallback khi AE không trả về đủ kết quả
        if len(answers) < num_pairs:
            print("[AE] Fallback: bổ sung bằng tách câu")
            extra = self._extract_answers_sentences(context, num_pairs * 2)
            seen_norm = {_nfc(a).lower() for a in answers}
            for e in extra:
                if _nfc(e).lower() not in seen_norm:
                    answers.append(e)
                    seen_norm.add(_nfc(e).lower())
            print(f"[AE] -> sau fallback: {len(answers)} đáp án")

        if not answers:
            print("[AE] Không tìm được đáp án nào.")
            return []

        # ── Bước 2: sinh câu hỏi cho từng đáp án ────────────
        pairs: List[Dict[str, str]] = []
        seen_q: set = set()

        for answer in answers:
            if len(pairs) >= num_pairs:
                break
            # Bỏ answer quá ngắn (1 ky tự, chỉ số, ...)
            if len(answer.strip()) < 2:
                print(f"[QG] Skip (quá ngắn): '{answer}'")
                continue
            print(f"[QG] Sinh câu hỏi cho answer: '{answer}'")
            question = self._generate_question(context, answer)
            if question and question not in seen_q:
                seen_q.add(question)
                pairs.append({"question": question, "answer": answer})
                print(f"[QG] -> Q: {question}")
            elif question:
                print(f"[QG] -> Skip (trùng): {question}")
            else:
                print(f"[QG] -> Không sinh được câu hỏi cho '{answer}'")

        print(f"[Generator] Tổng: {len(pairs)} cặp Q-A")
        return pairs


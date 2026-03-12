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
    pairs = gen.generate("Vấn đề bùng nổ về dữ liệu: khi....")
    # [{"question": "…", "answer": "…"}, …]
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT VÀ CẤU HÌNH BAN ĐẦU
# ═══════════════════════════════════════════════════════════════════════════════
# os: Thao tác với biến môi trường (lấy VIQAG_MODEL từ .env nếu có)
# re: Regex xử lý text (tách câu, loại dấu cách thừa)
# unicodedata: Normalize Unicode tiếng Việt (fix sai lệch NFC)
# logging: Ghi log cảnh báo/lỗi từ thư viện transformers
import os
import re
import unicodedata
import logging
import time
from typing import List, Dict, Optional

# Cấu hình logging: Chỉ show WARNING trở lên (ẩn các log INFO spam từ transformers)
logging.basicConfig(level=logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# HẰNG SỐ VÀ CẤU HÌNH MỨC CẦU CỪ 
# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT_MODEL: Multitask model (QG + AE trong 1 model). Model này:
#                - Extract answers (AE) từ context
#                - Generate questions (QG) cho từng answer
DEFAULT_MODEL = "shnl/vit5-vinewsqa-qg-ae"

# MAX_INPUT_LEN: Độ dài tối đa input cho tokenizer (512 tokens = ~300–400 từ)
#                Nếu vượt sẽ bị cắt bỏ (truncation=True)
MAX_INPUT_LEN = 512

# MAX_OUTPUT_LEN: Độ dài tối đa output sinh ra từ model (VD: "Big data" = 2 tokens)
MAX_OUTPUT_LEN = 128

# HL_TOKEN: Token đặc biệt để highlight phần cần attention của model
#           Format: "<hl> Big data <hl>" đưa model focus vào "Big data"
HL_TOKEN = "<hl>"


# ═══════════════════════════════════════════════════════════════════════════════
# HÀM TRỢ GIÚP (HELPERS)
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """
    Dọn dẹp text: Xóa dấu cách thừa, tab, newline từ việc model sinh ra.
    VD: "Big    data\n  là" → "Big data là"
    
    Params:
        text: Text cần dọn dẹp
    Returns:
        String đã chuẩn hóa (1 dấu cách giữa các từ, không dấu cách đầu/cuối)
    """
    return re.sub(r"\s+", " ", text).strip()


def _nfc(text: str) -> str:
    """
    Normalize Unicode NFC – fix lỗi so sánh chuỗi tiếng Việt khi diacritics.
    
    Vấn đề: Tiếng Việt có diacritics (dấu sắc, huyền, v.v.) có thể được encode 2 cách:
    - NFD (decomposed): "ế" = "e" + dấu = 2 ký tự
    - NFC (composed): "ế" = 1 ký tự
    
    Nếu model output NFC nhưng context là NFD (hoặc ngược lại) → so sánh sẽ fail.
    Solution: Convert cả 2 sang NFC rồi mới so sánh.
    
    Returns:
        Text đã normalize về NFC (chuẩn thống nhất)
    """
    return unicodedata.normalize("NFC", text)


def _answer_in_context(answer: str, context: str) -> bool:
    """
    Kiểm tra answer có trong context, chấp nhận sai lệch:
    - Unicode NFC/NFD khác nhau
    - Khoảng trắng thừa
    - Sai chính tả nhẹ từ model
    
    Chiến lược 2 cấp:
    1. Cấp 1 (Chính xác): Kiểm tra answer có substring trong context không
    2. Cấp 2 (Linh hoạt): Nếu lỗi → Kiểm tra >= 80% từ của answer xuất hiện trong context
       (chịu được model bỏ/sửa một vài từ)
    
    Params:
        answer: Đáp án cần kiểm tra (VD: "Big data")
        context: Văn bản gốc cần tìm trong đó
    
    Returns:
        True nếu answer hợp lệ (có trong context hoặc overlap >= 80%)
        False nếu answer sai hoặc không có trong context
    """
    a = _nfc(answer).lower()
    c = _nfc(context).lower()
    
    # Cấp 1: Kiểm tra if answer là substring của context (chính xác)
    if a in c:
        return True
    
    # Cấp 2: Kiểm tra word-level overlap
    # Lọc các từ >= 2 ký tự (loại từ 1 ký tự như "a", "ở" không meaningful)
    words = [w for w in a.split() if len(w) >= 2]
    if not words:
        return False
    
    # Tính % từ của answer có trong context
    # VD: answer="Big data model", context="Big data..."
    #     words=["big", "data", "model"], có 2/3 → 66% < 80% → reject
    return sum(1 for w in words if w in c) / len(words) >= 0.8
def _split_sentences(text: str) -> List[str]:
    """
    Tách text thành từng câu đơn lẻ (fallback cho AE khi model không work tốt).
    
    Cách tách:
    - Dùng regex lookbehind: Tách sau các dấu [.?!;] kèm whitespace
    - VD: "Câu 1. Câu 2! Câu 3" → ["Câu 1", "Câu 2", "Câu 3"]
    
    Filter:
    - Chỉ giữ các câu > 10 ký tự (loại câu quá ngắn)
    - Mỗi câu được strip() loại khoảng trắng đầu/cuối
    
    Returns:
        List câu tiếng Việt, mỗi câu > 10 ký tự
    """
    # Regex: (?<=[.?!;]) = positive lookbehind cho dấu kết thúc câu
    #        \s+ = 1 hoặc nhiều whitespace sau dấu kết thúc
    # Tác dụng: Tách text tại điểm kết thúc câu nhưng không xóa dấu kết thúc
    parts = re.split(r"(?<=[.?!;])\s+", text)
    
    # Filter: Chỉ giữ các phần > 10 ký tự (câu quá ngắn không useful)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _is_multitask(model_name: str) -> bool:
    """
    Detect xem model có hỗ trợ cả 2 task (QG + AE) hay chỉ 1 task (QG-only).
    
    Logic:
    - Multitask model: Tên chứa cả từ "qg" (Question Generation) 
                       VÀ "ae" (Answer Extraction)
    - Pipeline model: Chỉ "qg" hoặc chỉ "ae"
    
    Ví dụ:
    - "shnl/vit5-vinewsqa-qg-ae" → Tách: ["shnl", "vit5", "vinewsqa", "qg", "ae"]
      → Có cả "qg" và "ae" → True (multitask)
    - "namngo/pipeline-vit5-viquad-qg" → Tách: ["namngo", "pipeline", "vit5", "viquad", "qg"]
      → Chỉ có "qg" → False (pipeline QG-only)
    
    Params:
        model_name: Tên model HuggingFace hub (VD: "username/model-name-qg-ae")
    
    Returns:
        True nếu multitask (có cả QG + AE)
        False nếu pipeline (chỉ QG hoặc chỉ AE)
    """
    # Tách model_name theo dấu "-" hoặc "/" (convention của HF)
    # VD: "shnl/vit5-vinewsqa-qg-ae" → ["shnl", "vit5", "vinewsqa", "qg", "ae"]
    parts = re.split(r"[-/]", model_name.lower())
    
    # Check xem cả "qg" và "ae" đều có trong list parts
    return "qg" in parts and "ae" in parts


# ───────────────────────── class chính ──────────────────────────
class QAGenerator:
    """
    Sinh Question-Answer từ context tiếng Việt dùng ViT5 (local model).

    Tham số:
        model_name : Tên model HF hub.
        device     : "cpu" | "cuda" | "auto".
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
    ):
        """
        Khởi tạo QA Generator: Device, model_name, load model từ HuggingFace.
        
        Quy trình:
        1. Xác định model_name (từ param → biến env → default)
        2. Detect multitask hay pipeline
        3. Load model + tokenizer từ HuggingFace (lần đầu lâu 1-3 phút)
        
        Params:
            model_name: Tên model HF (VD: "shnl/vit5-vinewsqa-qg-ae")
                       Nếu None, sẽ lấy từ env var VIQAG_MODEL hoặc DEFAULT_MODEL
            device: "cuda" (GPU) | "cpu" (CPU) | "auto" (auto-detect)
        
        Raises:
            RuntimeError: Nếu thiếu thư viện torch/transformers
        """
        # Ưu tiên: param → biến env → hằng số DEFAULT_MODEL
        self.model_name = model_name or os.getenv("VIQAG_MODEL", DEFAULT_MODEL)
        self.device     = device
        
        # Detect: Model này hỗ trợ QG+AE hay chỉ QG?
        # multitask=True → Dùng 2-stage (AE → QG)
        # multitask=False → Dùng tách câu tĩnh thay AE
        self.multitask  = _is_multitask(self.model_name)

        # Khởi tạo None (sẽ gán trong _load_local())
        self._model      = None
        self._tokenizer  = None
        self._device_str = "cpu"

        # Gọi hàm tải model từ HuggingFace
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

        print(f"[Generator] Đang load model '{self.model_name}' (lần đầu ~1–3 phút)…")

        
        #khởi tạo và tải bộ tách từ (Tokenizer)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                legacy=False,
            )
        except Exception:
            # Fallback cho một số model ViT5/T5 khi AutoTokenizer không tương thích phiên bản mới.
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                legacy=False,
            )

        # Thêm special token <hl> vào bộ từ điển (vocabulary) của Tokenizernếu chưa có
        if HL_TOKEN not in self._tokenizer.get_vocab():
            self._tokenizer.add_special_tokens({"additional_special_tokens": [HL_TOKEN]})

        try:
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name
            )
        except Exception:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INFERENCE: Sinh text từ prompt (QA generation core)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _infer(self, prompt: str, max_new_tokens: int = MAX_OUTPUT_LEN,
               num_return_sequences: int = 1) -> List[str]:
        """
        Inference: Sinh 1 hoặc nhiều kết quả text từ prompt.
        
        Quy trình:
        1. Tokenize prompt → token IDs với max_length=512
        2. Move tensors lên GPU/CPU
        3. Model.generate(): Sinh token IDs dùng beam search
        4. Decode token IDs → text
        
        Beam Search: Dùng để tìm kết quả tốt nhất (không tham lam).
        - num_beams=4: Track 4 candidate sequences, chọn tối ưu nhất
        - Càng nhiều beams → Tốt hơn nhưng chậm hơn
        
        Params:
            prompt: Chuỗi prompt gủi cho model (VD: "extract answers: [context]")
            max_new_tokens: Độ dài tối đa output (default=128)
            num_return_sequences: Số output sinh ra (VD: 2 → sinh 2 answers khác nhau)
        
        Returns:
            List string: Danh sách kết quả (độ dài = num_return_sequences)
            VD: ["Big data", "Dữ liệu lớn"] với num_return_sequences=2
        """
        import torch
        
        # ─ Bước 1: Tokenize ─
        # Chuyển text prompt → tensor của token IDs
        # return_tensors="pt": PyTorch tensor format
        # max_length=512: Cắt nếu vượt 512 tokens
        # truncation=True: Cắt bỏ phần vừa
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(self._device_str)  # Move tensors sang GPU/CPU
        
        # ─ Bước 2: Config beam search ─
        # num_beams: Số lượng hypotheses theo dõi song song
        # Ít nhất 4 để diversity, nhiều hơn → compute tăng
        num_beams = max(4, num_return_sequences)
        
        # ─ Bước 3: Sinh token IDs (no_grad: tối ưu memory) ─
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Độ dài tối đa output
                num_beams=num_beams,            # Beam search
                num_return_sequences=num_return_sequences,  # Số output
                early_stopping=True,            # Dừng sớm khi tìm được solution tốt
            )
        
        # ─ Bước 4: Decode → text ─
        # Chuyển từng tensor token ID → string text
        # skip_special_tokens=True: Bỏ <hl>, <pad>, </s>, etc
        return [self._tokenizer.decode(i, skip_special_tokens=True) for i in ids]

    def _infer_one(self, prompt: str, max_new_tokens: int = MAX_OUTPUT_LEN) -> str:
        """
        Sinh 1 kết quả (wrapper của _infer()).
        
        Returns:
            String đầu tiên từ _infer (default num_return_sequences=1)
        """
        return self._infer(prompt, max_new_tokens)[0]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 1 (AE): EXTRACT ANSWERS từ context (Multitask model)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_answers_multitask(self, context: str, need: int) -> List[str]:
        """
        Answer Extraction (AE): Model trích xuất từng câu của context.
        Dùng multitask model để sinh candidates answer từ context.
        
        Quy trình:
        1. Tách context thành từng câu (dùng _split_sentences)
        2. Với mỗi câu: Highlight nó + gủi prompt "extract answers: [highlighted_context]"
        3. Model sinh đáp án candidate (có thể nhiều cái từ 1 câu)
        4. Filter: Chỉ giữ answers có trong context (tránh hallucination)
        5. Deduplicate: Loại trùng lặp
        6. Lặp đến khi có đủ answers
        
        Params:
            context: Đoạn văn cần trích xuất answers (VD: "Big data là...")
            need: Số answers cần (VD: 10). Hàm sinh ~2x cái này rồi filter
        
        Returns:
            List string: Danh sách answers được model rút ra từ context
            VD: ["Big data", "Dữ liệu lớn", "2010"]
        """
        # ─ Bước 0: Tách câu ─
        sentences = _split_sentences(context)
        answers: List[str] = []  # Lưu answers hợp lệ
        seen: set = set()         # Tracking answers đã thêm (loại dup)

        # ─ Tính số_candidates_per_câu ─
        # VD: need=5 câu, có 10 câu → seqs_per_sent=1 (1 answer/câu)
        #     need=10 câu, có 3 câu → seqs_per_sent=4 (4 answers/câu, max=4)
        seqs_per_sent = max(1, -(-need // max(len(sentences), 1)))  # ceil division
        seqs_per_sent = min(seqs_per_sent, 4)  # tối đa 4 (tránh OOM)

        # ─ Bước 1: Loop qua từng câu, sinh answers ─
        for sentence in sentences:
            # Early stop nếu đã có quá đủ candidates (need * 2)
            if len(answers) >= need * 2:
                break
            
            # Tìm vị trí câu trong context (để highlight)
            pos = context.find(sentence)
            if pos == -1:
                continue  # Câu không tìm thấy? Skip (không nên xảy ra)
            
            # ─ Bước 2: Highlight câu ─
            # Format: "[trước] <hl> [câu này] <hl> [sau]"
            # Token <hl> báo cho model: "Focus vào phần giữa <hl>...<hl>"
            highlighted = (
                context[:pos]
                + f"{HL_TOKEN} {sentence} {HL_TOKEN}"
                + context[pos + len(sentence):]
            )
            
            # ─ Bước 3: Gủi prompt "extract answers: [highlighted_context]" ─
            prompt = f"extract answers: {highlighted}"
            # Sinh seqs_per_sent candidates (VD: 2 answers từ câu này)
            raws = self._infer(prompt, max_new_tokens=128, num_return_sequences=seqs_per_sent)
            
            # ─ Bước 4: Filter + Deduplicate ─
            for raw in (_clean(r) for r in raws):
                # Normalize để so sánh (loại trùng)
                norm = _nfc(raw).lower()
                
                # Kiểm tra: Answer hợp lệ?
                if (raw                                      # Không rỗng
                    and len(raw) >= 2                        # >= 2 ký tự
                    and norm not in seen                     # Chưa thêm trước
                    and _answer_in_context(raw, context)):   # Có trong context
                    
                    # Thêm vào results
                    seen.add(norm)
                    answers.append(raw)
                    print(f"[AE] + '{raw}'")
                elif raw:
                    # Log lý do loại bỏ
                    print(f"[AE] drop '{raw}' (not in ctx or dup)")
        
        return answers

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 1 (FALLBACK): EXTRACT ANSWERS từ câu cố định
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_answers_sentences(self, context: str, num: int) -> List[str]:
        """
        Fallback AE: Nếu model AE không hoạt động tốt → dùng câu cố định.
        
        Chiến lược: Tách context thành từng câu (đơn giản, không dùng model).
        Ưu điểm: Luôn hoạt động, nhanh (không inference).
        Nhược: Answers có thể không phải ideal (là câu nguyên bản, không tóm tắt).
        
        This method được gọi trong generate() khi:
        - len(answers_từ_multitask) < num_pairs → Bổ sung thêm bằng sentences
        
        Params:
            context: Đoạn văn
            num: Số câu muốn lấy tối đa
        
        Returns:
            List string: Danh sách câu từ context (mỗi câu > 10 ký tự)
        """
        # Tách context theo câu, lấy tối đa num câu đầu
        return _split_sentences(context)[:num]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 2 (QG): GENERATE QUESTION cho mỗi answer
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _generate_question(self, context: str, answer: str) -> Optional[str]:
        """
        Question Generation (QG): Model sinh câu hỏi cho cặp (context, answer).
        
        Quy trình:
        1. Tìm vị trí answer trong context
        2. Highlight answer bằng token <hl>
        3. Gủi prompt "generate question: [highlighted_context]"
        4. Model sinh câu hỏi
        5. Clean + Return
        
        Params:
            context: Đoạn văn (VD: "Big data là tập hợp dữ liệu lớn")
            answer: Đáp án cần sinh câu hỏi cho nó (VD: "Big data")
        
        Returns:
            String: Câu hỏi được sinh ra (VD: "Big data là gì?")
            None: Nếu model không sinh ra gì (hiếm)
        """
        # ─ Tìm vị trí answer trong context ─
        # Normalize (NFC) và lowercase để so sánh
        pos = _nfc(context).lower().find(_nfc(answer).lower())
        
        # ─ Highlight answer ─
        if pos == -1:
            # Answer không tìm thấy trong context (rare) → Appending "answer" ở cuối
            highlighted = f"{context} {HL_TOKEN} {answer} {HL_TOKEN}"
        else:
            # Answer tìm thấy → Highlight vị trí đó
            # Lấy đoạn từ pos đến pos+len(answer) là phần answer
            highlighted = (
                context[:pos]
                + f"{HL_TOKEN} {context[pos:pos+len(answer)]} {HL_TOKEN}"
                + context[pos + len(answer):]
            )
        
        # ─ Sinh câu hỏi ─
        prompt = f"generate question: {highlighted}"
        q = _clean(self._infer_one(prompt))  # Sinh 1 câu hỏi
        
        # ─ Return ─
        return q if q else None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC API: MAIN PIPELINE - Sinh Q-A pairs từ context
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def generate(
        self,
        context: str,
        num_pairs: int = 5,
    ) -> List[Dict[str, str]]:
        """
        ★ HÀM CHÍNH: Sinh tối đa `num_pairs` cặp Q-A từ `context`.
        
        Two-stage pipeline:
        Stage 1 (AE): Extract answers từ context (dùng _extract_answers_multitask)
        Stage 2 (QG): Sinh question cho mỗi answer (dùng _generate_question)
        
        Quá trình:
        1. Clean context (xóa dấu cách thừa)
        2. Extract answers multitask → Nếu ko đủ → Fallback tách câu
        3. Loop qua từng answer: Sinh question + Dedup
        4. Return max num_pairs cặp Q-A
        
        Params:
            context: Đoạn văn tiếng Việt (VD: "Big data là tập hợp...")
                    Nên >= 50 ký tự để model hoạt động tốt
            num_pairs: Số cặp Q-A mong muốn (default=5)
        
        Returns:
            List[Dict]: [{"question": "...", "answer": "..."}, ...]
            VD:
            [
                {"question": "Big data là gì?", "answer": "Big data"},
                {"question": "Big data được định nghĩa là sao?", "answer": "Dữ liệu lớn"},
                ...
            ]
        """
        # ─ Chuẩn bị: Clean context ─
        context = _clean(context)
        if not context:
            return []  # Rỗng → Return []

        # ══════════════════════════════════════════════════════════════════════════════
        # STAGE 1: EXTRACT ANSWERS
        # ══════════════════════════════════════════════════════════════════════════════
        answers: List[str] = []
        if self.multitask:
            # Dùng multitask model để sinh answers
            num_sentences = len(_split_sentences(context))
            print(f"[AE] Trích xuất đáp án từ {len(context)} ky tự ({num_sentences} câu)...")
            
            # Sinh answers (multitask: model sinh từ context)
            answers = self._extract_answers_multitask(context, need=num_pairs * 2)
            print(f"[AE] -> {len(answers)} đáp án: {answers}")

        # ─ Fallback: Nếu AE không đủ → Bổ sung bằng sentences ─
        if len(answers) < num_pairs:
            print("[AE] Fallback: bổ sung bằng tách câu")
            # Tách câu làm answers thêm
            extra = self._extract_answers_sentences(context, num_pairs * 2)
            # Dedup: Chỉ thêm sentences chưa có trong answers
            seen_norm = {_nfc(a).lower() for a in answers}
            for e in extra:
                if _nfc(e).lower() not in seen_norm:
                    answers.append(e)
                    seen_norm.add(_nfc(e).lower())
            print(f"[AE] -> sau fallback: {len(answers)} đáp án")

        # ─ Check: Có answers không? ─
        if not answers:
            print("[AE] Không tìm được đáp án nào.")
            return []  # Không có answers → Fail

        # ══════════════════════════════════════════════════════════════════════════════
        # STAGE 2: GENERATE QUESTIONS
        # ══════════════════════════════════════════════════════════════════════════════
        pairs: List[Dict[str, str]] = []  # Lưu Q-A pairs cuối cùng
        seen_q: set = set()               # Tracking questions (loại dup)

        for answer in answers:
            # ─ Early stop: Đủ pairs rồi ─
            if len(pairs) >= num_pairs:
                break
            
            # ─ Filter: Answer quá ngắn (1 ký tự, chỉ số, etc) ─
            if len(answer.strip()) < 2:
                print(f"[QG] Skip (quá ngắn): '{answer}'")
                continue
            
            # ─ Sinh question ─
            print(f"[QG] Sinh câu hỏi cho answer: '{answer}'")
            question = self._generate_question(context, answer)
            
            # ─ Validate + Thêm vào results ─
            if question and question not in seen_q:
                # Question hợp lệ + chưa có → Thêm
                seen_q.add(question)
                pairs.append({"question": question, "answer": answer})
                print(f"[QG] -> Q: {question}")
            elif question:
                # Question có nhưng trùng lặp
                print(f"[QG] -> Skip (trùng): {question}")
            else:
                # Model không sinh được question cho answer này
                print(f"[QG] -> Không sinh được câu hỏi cho '{answer}'")

        # ─ Final log ─
        print(f"[Generator] Tổng: {len(pairs)} cặp Q-A")
        return pairs


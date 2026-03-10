"""
app.py  –  Giao diện Giáo Viên – Sinh MCQ Tiếng Việt
─────────────────────────────────────────────────────
Pipeline: Văn bản → ViQAG (ViT5) → Q+A → LLM → Distractors → MCQ

Chạy:
    streamlit run app.py
"""

import os, random, json, time, copy, base64
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# ── load .env ──────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ══════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hệ thống tạo câu hỏi trắc nghiệm",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Background image config
BG_IMAGE_PATH = Path(__file__).resolve().parent.parent / "assets" / "918414.jpg"
BG_IMAGE_OPACITY = 0.45  # 0.0 = an het anh, 1.0 = ro net anh


def _read_base64_image(path: Path) -> str:
    """Doc anh local va tra ve chuoi base64 de gan vao CSS."""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""


_bg_b64 = _read_base64_image(BG_IMAGE_PATH)
_bg_overlay = max(0.0, min(1.0, 1.0 - BG_IMAGE_OPACITY))

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── tổng thể ── */
[data-testid="stAppViewContainer"] { background: #f7f9fc; }
[data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid #e5e7eb; }

/* ── card câu hỏi ── */
.mcq-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 18px 22px 14px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.mcq-card.selected { border-left-color: #10b981; }
.mcq-q  { font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 10px; }
.opt    { font-size: .97rem; padding: 3px 0; color: #374151; }
.opt-correct { color: #16a34a; font-weight: 700; }

/* ── stage badge ── */
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: .78rem; font-weight: 600;
    margin-left: 6px;
}
.badge-viqag  { background:#dbeafe; color:#1d4ed8; }
.badge-llm    { background:#dcfce7; color:#15803d; }
.badge-done   { background:#fef9c3; color:#92400e; }

/* ── preview panel ── */
.preview-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 14px 16px; font-size: .9rem;
    white-space: pre-wrap; font-family: monospace;
}

/* ── stage info ── */
.stage-info {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 10px 14px;
    font-size: .88rem; color: #1e40af; margin-bottom: 12px;
}

/* ── button full-width ── */
div[data-testid="stButton"] > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

if _bg_b64:
    st.markdown(
        f"""
<style>
[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(
            rgba(247, 249, 252, {_bg_overlay:.3f}),
            rgba(247, 249, 252, {_bg_overlay:.3f})
        ),
        url("data:image/jpeg;base64,{_bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# Constants & helpers
# ══════════════════════════════════════════════════════════════
LABELS  = ["A", "B", "C", "D", "E"]
EXAMPLE = (
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam. "
    "Đây là thành phố lớn thứ hai cả nước về dân số sau Thành phố Hồ Chí Minh. "
    "Hà Nội nằm ở vùng đồng bằng sông Hồng, cách bờ biển khoảng 120 km về phía tây. "
    "Thành phố có lịch sử hơn 1000 năm và đã từng được gọi là Thăng Long dưới triều đại nhà Lý. "
    "Dân số Hà Nội hiện nay khoảng 8,4 triệu người. "
    "Hà Nội có nhiều trường đại học lớn như Đại học Quốc gia Hà Nội, Đại học Bách Khoa. "
    "Sông Hồng chảy qua phía đông bắc thành phố, tạo nên vùng đất màu mỡ phù sa."
)


def _generate_example_paragraph() -> str:
    """Đoạn văn ví dụ mặc định"""
    return EXAMPLE

def _init_state():
    defaults = {
        "mcq_list":    [],      # List[Dict] – danh sách câu sau edit
        "history":     [],      # List[List[Dict]] – lịch sử 10 lần gần nhất
        "context_buf": "",      # context hiện tại
        "selected":    set(),   # index câu được chọn để export
        "regen_idx":   None,    # index câu cần regenerate
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def build_mcq(question: str, answer: str, distractors: List[str]) -> Dict:
    options = distractors[:3] + [answer]
    random.shuffle(options)
    return {
        "question":      question,
        "answer":        answer,
        "options":       options,
        "correct_label": LABELS[options.index(answer)],
        "source":        "ViT5 + LLM",
    }


def shuffle_mcq(mcq: Dict) -> Dict:
    """Xáo lại thứ tự đáp án, giữ nguyên đáp án đúng."""
    m = copy.deepcopy(mcq)
    random.shuffle(m["options"])
    m["correct_label"] = LABELS[m["options"].index(m["answer"])]
    return m


def mcq_to_text(mcq_list: List[Dict], show_ans: bool = False) -> str:
    """Render danh sách MCQ thành text"""
    lines = []
    for i, m in enumerate(mcq_list, 1):
        lines.append(f"Câu {i}. {m['question']}")
        for j, opt in enumerate(m["options"]):
            lbl  = LABELS[j]
            mark = "  ← ĐÁP ÁN" if (show_ans and lbl == m["correct_label"]) else ""
            lines.append(f"   {lbl}. {opt}{mark}")
        lines.append("")
    return "\n".join(lines)


def save_to_history(mcq_list: List[Dict]):
    """Đẩy đề mới lên đầu lịch sử, giữ tối đa 10 đề."""
    hist = st.session_state["history"]
    hist.insert(0, copy.deepcopy(mcq_list))
    st.session_state["history"] = hist[:10]


# ══════════════════════════════════════════════════════════════
# Cached model loaders
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_qa_generator(model_name: str, use_api: bool, hf_token: str):
    from generator import QAGenerator
    clean_model_name = (model_name or "").strip() or None
    clean_hf_token = (hf_token or "").strip() or None
    return QAGenerator(model_name=clean_model_name, use_api=use_api, hf_token=clean_hf_token)


@st.cache_resource(show_spinner=False)
def load_distractor_generator(backend: str, model: str, api_key: str, ollama_host: str):
    from distractor import DistractorGenerator
    return DistractorGenerator(
        backend=backend or None,
        model=model or None,
        api_key=api_key or None,
        ollama_host=ollama_host or "http://localhost:11434",
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR – Cấu hình giáo viên
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Tùy chỉnh")

    # ── ViQAG ──────────────────────────────────────────────────
    with st.expander("Stage 1 – ViT5 (sinh Q-A)", expanded=True):
        qa_mode = st.selectbox(
            "Phương thức",
            ["Local model (~1GB RAM)", "HF Inference API (không cần GPU)"],
            index=0,
            help="Local model: tự download về máy, không cần token. API: gọi HF.",
        )
        use_api_flag = "API" in qa_mode #Biến boolean để biết có đang chọn mode API không
        viqag_model  = st.text_input(
            "Model",
            value=os.getenv("VIQAG_MODEL", "shnl/vit5-vinewsqa-qg-ae"),
        )
        hf_token_input = st.text_input(
            "HF Token",
            value=os.getenv("HF_TOKEN", ""),
            type="password",
            help="Chỉ cần khi dùng HF Inference API. Lấy tại huggingface.co/settings/tokens",
        )
        if use_api_flag:
            st.warning(
                "HF Inference API chỉ hỗ trợ các model phổ biến được HF featured. "
                "Các model ViT5 tiếng Việt thường không khả dụng qua API → khuyến nghị dùng **Local model**.",
                icon="⚠️",
            )

    # ── LLM ────────────────────────────────────────────────────
    with st.expander("🤖 Stage 2 – LLM (sinh distractors)", expanded=True):
        llm_backend = st.selectbox(
            "Nguồn LLM",
            ["auto", "gemini", "openai", "ollama"],
            index=0,
        )
        if llm_backend != "auto":
            llm_model = st.text_input(
                "Model name",
                value="",
                placeholder="Để trống = mặc định",
                help="VD: gemini-1.5-flash, gpt-4o-mini, llama3",
            )
            if llm_backend != "ollama":
                llm_api_key = st.text_input(
                    "API Key",
                    value=os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
                    type="password",
                )
            else:
                llm_api_key = None
            if llm_backend == "ollama":
                ollama_host = st.text_input(
                    "Ollama Host",
                    value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                )
            else:
                ollama_host = None
    st.divider()

    # ── Tuỳ chọn đề ────────────────────────────────────────────
    st.markdown("### Lịch sử")
    num_pairs       = st.slider("Số câu hỏi muốn sinh", 2, 15, 5)
    num_distractors = st.radio("Số đáp án sai / câu", [3, 4], index=0, horizontal=True)
    difficulty      = st.select_slider(
        "Độ khó distractor",
        options=["Dễ", "Trung bình", "Khó"],
        value="Trung bình",
        help="Ảnh hưởng đến độ nhiễu trong prompt LLM",
    )

    st.divider()

    # ── Lịch sử ────────────────────────────────────────────────
    hist = st.session_state["history"]
    if hist:
        st.markdown(f"### 🕑 Lịch sử ({len(hist)} đề)")
        for i, h in enumerate(hist):
            if st.button(f"Đề #{i+1}  –  {len(h)} câu", key=f"hist_{i}"):
                st.session_state["mcq_list"] = copy.deepcopy(h)
                st.rerun()

    st.divider()
    st.caption("ViT5 + LLM · Vietnamese MCQ Generator")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(
    "# 📝 Hệ thống tạo câu hỏi trắc nghiệm"
)

# ══════════════════════════════════════════════════════════════
# LAYOUT: 2 cột chính  (input+output) + preview
# ══════════════════════════════════════════════════════════════
col_main, col_preview = st.columns([3, 2], gap="large")

# ──────────────────────────────────────────────────────────────
# CỘT TRÁI – Input & Generate
# ──────────────────────────────────────────────────────────────
with col_main:

    # ── Text input ─────────────────────────────────────────────
    st.markdown("#### 📄 Bước 1 – Nhập đoạn văn bản")

    col_ta, col_ex = st.columns([6, 1])
    with col_ex:
        st.markdown(" ")   # padding
        if st.button("🎲\nVí dụ", use_container_width=True, help="Văn mẫu sẵn có"):
            with st.spinner("Đang lấy văn mẫu…"):
                st.session_state["context_input"] = _generate_example_paragraph()
            st.rerun()
    with col_ta:
        context = st.text_area(
            "Nội dung",
            height=200,
            placeholder="Dán đoạn văn từ sách giáo khoa, bài báo, bài giảng…",
            label_visibility="collapsed",
            key="context_input",
        )

    word_count = len(context.split()) if context else 0
    st.caption(f"Số từ: **{word_count}**")

    # ── Nút Generate ───────────────────────────────────────────
    st.markdown("#### 🚀 Bước 2 – Sinh câu hỏi")
    num_pairs = st.number_input(
        "Số câu hỏi muốn sinh",
        min_value=1,
        max_value=20,
        value=num_pairs,
        step=1,
        help="Số cặp Q-A tối đa sẽ được sinh ra từ đoạn văn",
    )
    generate_btn = st.button(
        "✨ Sinh câu hỏi trắc nghiệm",
        type="primary",
        use_container_width=True,
    )

    # ══════════════════════════════════════════════════════════
    # PIPELINE
    # ══════════════════════════════════════════════════════════
    if generate_btn:
        if not context or len(context.strip()) < 20:
            st.warning("⚠️ Đoạn văn quá ngắn. Nhập ít nhất 20 ký tự.")
        else:
            ctx = context.strip()
            st.session_state["context_buf"] = ctx
            mcq_list_new = []

            # difficulty → prompt modifier
            diff_hint = {
                "Dễ": "gần giống nhau về âm thanh hoặc cấu trúc, số từ gần bằng nhau",
                "Trung bình": "hợp lý và có sức nhiễu cao, số từ gần bằng nhau",
                "Khó": "rất dễ gây nhầm lẫn, tương tự đáp án đúng về ngữ nghĩa, số từ gần bằng nhau",
            }[difficulty]

            # ── Load models ───────────────────────────────────
            with st.status("⚙️ Load model…", expanded=True) as s:
                try:
                    st.write("🔄 Khởi động ViT5 (sinh câu hỏi)...")
                    qa_gen = load_qa_generator(viqag_model, use_api_flag, hf_token_input)
                    st.write("🔄 Đang kết nối đến LLM ...")
                    dist_gen = load_distractor_generator(
                        None if llm_backend == "auto" else llm_backend,
                        llm_model or None,
                        llm_api_key or None,
                        ollama_host,
                    )
                    s.update(label="✅ Kết nối thành công!", state="complete")
                except Exception as e:
                    s.update(label="❌ Lỗi kết nối", state="error")
                    st.error(str(e))
                    st.stop()

            # ── Stage 1: ViQAG ────────────────────────────────
            with st.status(
                " **Stage 1 · ViT5** – Đang phân tích văn bản và sinh Q-A…",
                expanded=True,
            ) as s:
                try:
                    # Yêu cầu thêm 50% để bù cho lọc chất lượng (trùng lặp, answer leakage…)
                    _request_pairs = max(num_pairs + 2, int(num_pairs * 1.5))
                    qa_pairs = qa_gen.generate(ctx, num_pairs=_request_pairs)
                    if not qa_pairs:
                        s.update(label="⚠️ ViT5 không trả về kết quả", state="error")
                        st.warning(
                            "Không tìm thấy câu hỏi phù hợp.  \n"
                            "Thử: văn bản dài hơn · kiểm tra HF Token · đổi sang Local model."
                        )
                        st.stop()
                    s.update(
                        label=f"✅ Stage 1 hoàn tất – {len(qa_pairs)} cặp Q-A",
                        state="complete",
                    )
                except Exception as e:
                    s.update(label="❌ Lỗi ViT5", state="error")
                    st.error(f"ViT5 lỗi: {e}")
                    st.stop()

            # ── Quality filter Q-A pairs ───────────────────
            def _jaccard(q1: str, q2: str) -> float:
                w1 = set(q1.lower().split())
                w2 = set(q2.lower().split())
                if not w1 or not w2:
                    return 0.0
                return len(w1 & w2) / len(w1 | w2)

            def _filter_qa(pairs):
                seen_q = []
                result = []
                for p in pairs:
                    
                    q_ = p.get("question", "").strip()
                    a_ = p.get("answer", "").strip()

                    if not q_ or not a_:
                        continue
                    
                    # Bỏ answer quá ngắn
                    if len(a_.split()) < 1:
                        continue
                    # Bỏ answer quá dài (>35 từ)
                    if len(a_.split()) > 30:
                        continue
                    # Bỏ answer xuất hiện trong question (answer leakage)
                    if a_.lower() in q_.lower():
                        continue
                    # Bỏ câu hỏi gần giống nhau – Jaccard >= 0.40
                    if any(_jaccard(q_, sq) >= 0.40 for sq in seen_q):
                        continue
                    seen_q.append(q_)
                    result.append(p)
                return result

            qa_pairs = _filter_qa(qa_pairs)[:num_pairs]  # giữ tối đa num_pairs sau lọc
            if not qa_pairs:
                st.warning("⚠️ Sau khi lọc chất lượng, không còn câu hỏi phù hợp. Thử đoạn văn khác.")
                st.stop()

            # ── Stage 2: LLM distractors ─────────────────────
            progress_bar = st.progress(0, text="🤖 Stage 2 · LLM – Đang sinh distractors…")
            errors_list  = []

            for i, pair in enumerate(qa_pairs):
                q, a = pair["question"], pair["answer"]
                progress_bar.progress(
                    i / len(qa_pairs),
                    text=f"🤖 Stage 2 · LLM – Câu {i+1}/{len(qa_pairs)}: {q[:45]}…",
                )
                # Throttle: ≥4s between calls to stay within Gemini free-tier rate limit
                if i > 0:
                    time.sleep(1)
                try:
                    augmented_ctx = f"{ctx}\n[Yêu cầu: distractors {diff_hint}]"
                    distractors = dist_gen.generate(
                        question=q, answer=a,
                        context=augmented_ctx,
                        num_distractors=num_distractors,
                    )
                    while len(distractors) < num_distractors:
                        distractors.append(f"[Đáp án sai {len(distractors)+1}]")
                    mcq_list_new.append(build_mcq(q, a, distractors[:num_distractors]))
                except Exception as e:
                    print(f"[Distractor] FAILED câu {i+1}: {e}")
                    errors_list.append(f"Câu {i+1}: {e}")
                    # Fallback: vẫn thêm câu với placeholder distractor
                    placeholders = [f"[Đáp án sai {j+1}]" for j in range(num_distractors)]
                    mcq_list_new.append(build_mcq(q, a, placeholders))

            progress_bar.progress(1.0, text="✅ Stage 2 hoàn tất!")
            time.sleep(0.3)
            progress_bar.empty()

            if errors_list:
                st.warning(
                    f"⚠️ LLM distractor gặp lỗi {len(errors_list)}/{len(qa_pairs)} câu "
                    f"(dùng placeholder tạm). "
                    f"Lỗi: {str(errors_list[0])[:120]}"
                )

            st.session_state["mcq_list"] = mcq_list_new
            st.session_state["selected"] = set(range(len(mcq_list_new)))
            save_to_history(mcq_list_new)
            if mcq_list_new:
                st.success(f"🎉 Hoàn tất! Đã tạo **{len(mcq_list_new)} câu trắc nghiệm**.")
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # DANH SÁCH CÂU HỎI – chỉnh sửa / xóa / regenerate
    # ══════════════════════════════════════════════════════════
    mcq_list: List[Dict] = st.session_state["mcq_list"]

    if mcq_list:
        st.divider()
        st.markdown(f"#### ✏️ Bước 3 – Xem và chỉnh sửa ({len(mcq_list)} câu)")

        # Thanh hành động nhanh
        act1, act2, act3, act4 = st.columns(4)
        with act1:
            if st.button("☑️ Chọn tất cả"):
                st.session_state["selected"] = set(range(len(mcq_list)))
                st.rerun()
        with act2:
            if st.button("☐ Bỏ chọn tất cả"):
                st.session_state["selected"] = set()
                st.rerun()
        with act3:
            if st.button("🔀 Xáo đáp án"):
                st.session_state["mcq_list"] = [shuffle_mcq(m) for m in mcq_list]
                st.rerun()
        with act4:
            if st.button("➕ Thêm câu trống"):
                st.session_state["mcq_list"].append({
                    "question": "Câu hỏi mới…",
                    "answer": "Đáp án đúng",
                    "options": ["Đáp án đúng", "Sai 1", "Sai 2", "Sai 3"],
                    "correct_label": "A",
                    "source": "thủ công",
                })
                st.session_state["selected"].add(len(st.session_state["mcq_list"]) - 1)
                st.rerun()

        st.markdown("")

        # ── Render từng card ───────────────────────────────────
        to_delete = None

        for idx, mcq in enumerate(mcq_list):
            is_selected = idx in st.session_state["selected"]
            card_cls    = "mcq-card selected" if is_selected else "mcq-card"

            with st.container():
                # Header card: checkbox + số câu + nguồn
                hc1, hc2 = st.columns([1, 11])
                with hc1:
                    checked = st.checkbox(
                        f"Chọn câu {idx+1}", value=is_selected,
                        key=f"sel_{idx}",
                        label_visibility="collapsed",
                    )
                    if checked != is_selected:
                        if checked:
                            st.session_state["selected"].add(idx)
                        else:
                            st.session_state["selected"].discard(idx)
                        st.rerun()

                with hc2:
                    st.markdown(
                        f'<span style="font-weight:700;font-size:1rem;">Câu {idx+1}</span>'
                        f'<span class="badge badge-viqag" style="font-size:.7rem;">{mcq.get("source","ViT5+LLM")}</span>',
                        unsafe_allow_html=True,
                    )

                # Nội dung câu hỏi (có thể chỉnh)
                with st.expander(f"📝 {mcq['question'][:80]}{'…' if len(mcq['question'])>80 else ''}", expanded=False):
                    new_q = st.text_input(
                        "Nội dung câu hỏi",
                        value=mcq["question"],
                        key=f"q_{idx}",
                    )
                    if new_q != mcq["question"]:
                        st.session_state["mcq_list"][idx]["question"] = new_q
                        st.rerun()

                    st.markdown("**Các lựa chọn** (xanh = đáp án đúng):")
                    for j, opt in enumerate(mcq["options"]):
                        lbl        = LABELS[j]
                        is_correct = (lbl == mcq["correct_label"])
                        c1, c2, c3 = st.columns([1, 8, 2])
                        with c1:
                            st.markdown(
                                f'<span style="color:{"#16a34a" if is_correct else "#6b7280"};'
                                f'font-weight:{"700" if is_correct else "400"}">{lbl}.</span>',
                                unsafe_allow_html=True,
                            )
                        with c2:
                            new_opt = st.text_input(
                                f"opt_{idx}_{j}",
                                value=opt,
                                key=f"opt_{idx}_{j}",
                                label_visibility="collapsed",
                            )
                            if new_opt != opt:
                                old_ans = mcq["answer"]
                                st.session_state["mcq_list"][idx]["options"][j] = new_opt
                                if is_correct:
                                    st.session_state["mcq_list"][idx]["answer"] = new_opt
                                st.rerun()
                        with c3:
                            if not is_correct:
                                if st.button("✔ Đặt đúng", key=f"setcorrect_{idx}_{j}"):
                                    st.session_state["mcq_list"][idx]["correct_label"] = lbl
                                    st.session_state["mcq_list"][idx]["answer"] = opt
                                    st.rerun()

                    # Nút hành động câu
                    rb1, rb2, rb3 = st.columns(3)
                    with rb1:
                        if st.button("🔀 Xáo lại câu này", key=f"shuffle_{idx}"):
                            st.session_state["mcq_list"][idx] = shuffle_mcq(
                                st.session_state["mcq_list"][idx]
                            )
                            st.rerun()
                    with rb2:
                        if st.button("🔁 Regenerate distractors", key=f"regen_{idx}"):
                            st.session_state["regen_idx"] = idx
                            st.rerun()
                    with rb3:
                        if st.button("🗑️ Xóa câu này", key=f"del_{idx}", type="secondary"):
                            to_delete = idx

        # Xóa câu (sau vòng lặp để tránh lỗi index)
        if to_delete is not None:
            st.session_state["mcq_list"].pop(to_delete)
            st.session_state["selected"].discard(to_delete)
            st.session_state["selected"] = {
                i if i < to_delete else i - 1
                for i in st.session_state["selected"] if i != to_delete
            }
            st.rerun()

        # ── Regenerate single question ────────────────────────
        regen_idx = st.session_state.get("regen_idx")
        if regen_idx is not None and 0 <= regen_idx < len(mcq_list):
            st.session_state["regen_idx"] = None
            with st.spinner(f"🔁 Đang regenerate distractors câu {regen_idx+1}…"):
                try:
                    dist_gen = load_distractor_generator(
                        None if llm_backend == "auto" else llm_backend,
                        llm_model or None,
                        llm_api_key or None,
                        ollama_host,
                    )
                    m = mcq_list[regen_idx]
                    new_d = dist_gen.generate(
                        question=m["question"], answer=m["answer"],
                        context=st.session_state.get("context_buf", ""),
                        num_distractors=num_distractors,
                    )
                    while len(new_d) < num_distractors:
                        new_d.append(f"[Đáp án sai {len(new_d)+1}]")
                    st.session_state["mcq_list"][regen_idx] = build_mcq(
                        m["question"], m["answer"], new_d[:num_distractors]
                    )
                    st.success(f"✅ Đã cập nhật câu {regen_idx+1}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi regenerate: {e}")

    else:
        # Màn chào ban đầu
        st.info(
            "**Hướng dẫn nhanh:**\n"
            "1. Dán đoạn văn bản vào ô phía trên\n"
            "2. Bấm **Sinh câu hỏi trắc nghiệm**\n"
            "3. Chỉnh sửa câu hỏi nếu cần\n"
            "4. Export Word/PDF từ cột bên phải"
        )
        with st.expander("ℹ️ Về pipeline ViT5 + LLM"):
            st.markdown("""
| Thành phần | Vai trò |
|---|---|
| **ViT5** | Model fine-tuned `shnl/vit5-vinewsqa-qg-ae` sinh câu hỏi & trích đáp án từ văn bản tiếng Việt |
| **LLM** | Sinh 3–4 đáp án nhiễu (Groq / Gemini / OpenAI / Ollama) |
| **MCQ Builder** | Xáo trộn đáp án, đánh nhãn A/B/C/D |
            """)


# ──────────────────────────────────────────────────────────────
# CỘT PHẢI – Preview & Export
# ──────────────────────────────────────────────────────────────
with col_preview:
    mcq_list: List[Dict] = st.session_state["mcq_list"]
    selected: set        = st.session_state["selected"]
    export_list          = [mcq_list[i] for i in sorted(selected) if i < len(mcq_list)]

    st.markdown("#### 👁️ Xem trước & Xuất đề")

    if export_list:
        st.caption(f"**{len(export_list)}/{len(mcq_list)}** câu được chọn để xuất")

        # Preview text
        preview_txt = mcq_to_text(export_list, show_ans=False)
        st.markdown(
            f'<div class="preview-box">{preview_txt}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### 📤 Xuất đề")

        # ── JSON ───────────────────────────────────────────────
        st.download_button(
            "⬇️ JSON (import Quizizz / Google Forms)",
            data=json.dumps(export_list, ensure_ascii=False, indent=2),
            file_name="mcq.json",
            mime="application/json",
            use_container_width=True,
        )

        # ── TXT ────────────────────────────────────────────────
        st.download_button(
            "⬇️ TXT (kèm đáp án)",
            data=mcq_to_text(export_list, show_ans=True),
            file_name="mcq.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ── Word ───────────────────────────────────────────────
        try:
            from export_utils import export_word_bytes
            word_bytes = export_word_bytes(export_list)
            st.download_button(
                "⬇️ Word (.docx) – có đáp án trang sau",
                data=word_bytes,
                file_name="mcq.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Word: {e}")

        # ── PDF ────────────────────────────────────────────────
        try:
            from export_utils import export_pdf_bytes
            pdf_bytes = export_pdf_bytes(export_list)
            st.download_button(
                "⬇️ PDF – có đáp án trang sau",
                data=pdf_bytes,
                file_name="mcq.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF: {e}")

        st.divider()

        # ── Đáp án riêng ───────────────────────────────────────
        st.markdown("##### 🔑 Bảng đáp án")
        ans_md = "| Câu | Đáp án |\n|---|---|\n"
        for i, m in enumerate(export_list, 1):
            ans_md += f"| {i} | **{m['correct_label']}**. {m['answer']} |\n"
        st.markdown(ans_md)

    else:
        st.info("Chọn câu hỏi ở cột trái (checkbox ☑) để xem preview và xuất đề.")

        if mcq_list:
            st.markdown(f"Hiện có **{len(mcq_list)} câu** chưa được chọn.")
            if st.button("☑️ Chọn tất cả & preview"):
                st.session_state["selected"] = set(range(len(mcq_list)))
                st.rerun()


# ══════════════════════════════════════════════════════════════
# CSS tùy chỉnh
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .mcq-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .mcq-question { font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; color: #1a202c; }
    .opt-correct  { color: #16a34a; font-weight: 600; }
    .opt-wrong    { color: #374151; }
    .badge-backend {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #dbeafe;
        color: #1d4ed8;
        margin-left: 6px;
    }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Singleton cache cho model (chỉ load 1 lần trong session)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_qa_generator(model_name: str, use_api: bool, hf_token: str):
    from generator import QAGenerator
    clean_model_name = (model_name or "").strip() or None
    clean_hf_token = (hf_token or "").strip() or None
    return QAGenerator(model_name=clean_model_name, use_api=use_api, hf_token=clean_hf_token)


@st.cache_resource(show_spinner=False)
def load_distractor_generator(backend: str, model: str, api_key: str, ollama_host: str):
    from distractor import DistractorGenerator
    return DistractorGenerator(
        backend=backend or None,
        model=model or None,
        api_key=api_key or None,
        ollama_host=ollama_host or "http://localhost:11434",
    )

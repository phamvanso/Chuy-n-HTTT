"""
app.py  –  Giao diện Giáo Viên – Sinh MCQ Tiếng Việt
─────────────────────────────────────────────────────
Pipeline: Văn bản → ViQAG (ViT5) → Q+A → LLM → Distractors → MCQ

Chạy:
    streamlit run app.py
"""

import os, random, json, time, copy
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── tổng thể ── */
[data-testid="stSidebar"]          { border-right: 1px solid rgba(128, 128, 128, 0.25); }

/* ── card câu hỏi ── */
.mcq-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 18px 22px 14px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
.mcq-card.selected { border-left-color: #10b981; }
.mcq-q  { font-size: 1.05rem; font-weight: 700; color: var(--text-color); margin-bottom: 10px; }
.opt    { font-size: .97rem; padding: 3px 0; color: var(--text-color); }
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
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 8px; padding: 14px 16px; font-size: .9rem;
    white-space: pre-wrap; font-family: monospace;
}

/* ── stage info ── */
.stage-info {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 8px; padding: 10px 14px;
    font-size: .88rem; color: var(--text-color); margin-bottom: 12px;
}

/* ── button full-width ── */
div[data-testid="stButton"] > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Constants & helpers
# ══════════════════════════════════════════════════════════════
LABELS  = ["A", "B", "C", "D"]
EXAMPLE = (
    "Vấn đề bùng nổ về dữ liệu: khi các công cụ thu thập dữ liệu tự động và công nghệ về cơ sở dữ liệu đã trở nên hoàn thiện, một lượng lớn dữ liệu được thu thập và lưu trữ trong các cơ sở dữ liệu, kho dữ liệu và các kho lưu trữ thông tin khác."
    "Lúc này, chúng ta đang có quá nhiều dữ liệu nhưng chưa mang tính phục vụ có mục đích cho người sử dụng."
    "Chúng ta đang thiếu tri thức, tức là dữ liệu đã qua xử lý và phục vụ riêng cho mục đích của người sử dụng."
    "Vấn đề đặt ra là làm thế nào để khai thác tri thức từ khối dữ liệu khổng lồ hiện đang có."
    "Giải pháp cho việc khai phá tri thức chính là sự ra đời của công nghệ kho dữ liệu và các phương pháp khai phá dữ liệu."
    "Giải pháp này liên quan đến việc xây dựng kho dữ liệu lớn và các phương thức xử lý phân tích trực tuyến."
    "Một mục tiêu quan trọng là trích lọc ra tri thức có ích cho con người như các luật, mẫu, và các ràng buộc từ khối lượng lớn dữ liệu của một hay nhiều cơ sở dữ liệu."
    "Có nhiều lý do khiến việc khai phá dữ liệu trở nên cần thiết trong lĩnh vực thương mại."
    "Trong thế giới thực, rất nhiều dữ liệu đã được thu thập và lưu trữ có hệ thống trong các kho dữ liệu."
    "Các loại dữ liệu này bao gồm dữ liệu trên web và dữ liệu thương mại điện tử."
    "Ngoài ra còn có dữ liệu mua bán tại các cửa hàng và siêu thị."
)


def _generate_example_paragraph() -> str:
    """Đoạn văn ví dụ mặc định"""
    return EXAMPLE

def _init_state():
    """Khởi tạo session state cho Streamlit app.
    
    Session state lưu trữ:
    - mcq_list: Danh sách câu hỏi hiện tại (sau khi user edit)
    - history: Lịch sử 10 đề gần nhất (để load lại)
    - context_buf: Đoạn văn nguồn hiện tại (dùng cho regenerate)
    - selected: Set các index câu được chọn để export
    - regen_idx: Index câu đang được regenerate (None nếu không có)
    """
    defaults = {
        "mcq_list":    [],     
        "history":     [],     
        "context_buf": "",    
        "selected":    set(), 
        "regen_idx":   None,
        "editor_version": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def build_mcq(question: str, answer: str, distractors: List[str]) -> Dict:
    """Xây dựng câu MCQ từ question, answer và 3 distractors.
    
    Args:
        question: Câu hỏi
        answer: Đáp án đúng
        distractors: Danh sách đáp án sai (lấy 3 đầu tiên)
    
    Returns:
        Dict chứa: question, answer, options (đã xáo), correct_label (A/B/C/D), source
    
    Logic:
    1. Ghép 3 distractors + answer thành 4 options
    2. Shuffle ngẫu nhiên thứ tự options
    3. Tìm vị trí answer trong options → gán label A/B/C/D
    """
    options = distractors[:3] + [answer]
    random.shuffle(options)
    return {
        "question":      question,
        "answer":        answer,
        "options":       options,
        "correct_label": LABELS[options.index(answer)],
        "source":        "ViT5 + LLM",
    }


def mcq_to_text(mcq_list: List[Dict], show_ans: bool = False) -> str:
    """Render danh sách MCQ thành text thuần (cho preview/export TXT).
    
    Args:
        mcq_list: Danh sách các câu hỏi
        show_ans: Có hiển thị đáp án đúng hay không
    
    Returns:
        String text format:
        Câu 1. [question]
           A. [option]
           B. [option]  ← ĐÁP ÁN (nếu show_ans=True)
           ...
    """
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
    """Đẩy đề mới lên đầu lịch sử, giữ tối đa 10 đề.
    
    Args:
        mcq_list: Danh sách câu hỏi vừa sinh
    
    Logic:
    1. Deep copy để tránh ảnh hưởng khi user edit sau
    2. Insert vào đầu mảng history
    3. Slice [:10] để chỉ giữ 10 đề gần nhất
    """
    hist = st.session_state["history"]
    hist.insert(0, copy.deepcopy(mcq_list))
    st.session_state["history"] = hist[:10]


# ══════════════════════════════════════════════════════════════
# Cached model loaders (Local only)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_qa_generator(model_name: str):
    """Load ViT5 model cho Question-Answer Generation (cached).
    
    Args:
        model_name: Tên model HuggingFace ( shnl/vit5-vinewsqa-qg-ae)
    
    Returns:
        QAGenerator instance
    
    Caching:
    - @st.cache_resource: Load 1 lần duy nhất, tái sử dụng cross-session
    - Model được load vào RAM/GPU và giữ nguyên suốt app chạy
    - Reload chỉ khi model_name thay đổi hoặc restart app
    """
    from generator import QAGenerator
    clean_model_name = (model_name or "").strip() or None
    return QAGenerator(model_name=clean_model_name)


@st.cache_resource(show_spinner=False)
def load_distractor_generator(ollama_host: str):
    """Khởi tạo Ollama LLM client cho distractor generation (cached).
    
    Args:
        ollama_host: URL Ollama server (vd: http://localhost:11434)
    
    Returns:
        DistractorGenerator instance kết nối Ollama
    
    Caching:
    - Client connection được cache, tránh reconnect mỗi lần generate
    - Model mặc định: qwen2.5:7b (có thể thay bằng llama3/gemma)
    """
    from distractor import DistractorGenerator
    return DistractorGenerator(
        model="qwen2.5:7b",
        ollama_host=ollama_host or "http://localhost:11434",
    )


# ══════════════════════════════════════════════════════════════
# CẤU HÌNH MẶC ĐỊNH (Local Only - Ollama + ViT5)
# ══════════════════════════════════════════════════════════════
# Đọc cấu hình từ file .env
# ── ViQAG config ──────────────────────────────────────
viqag_model = os.getenv("VIQAG_MODEL", "shnl/vit5-vinewsqa-qg-ae")  # Model HuggingFace cho QG+AE

# ── LLM config ────────────────────────────────────────────────
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Ollama API endpoint

# ── Tuỳ chọn đề ────────────────────────────────────────
num_pairs = 5  # Số câu hỏi mặc định (user có thể thay đổi trong UI)
num_distractors = 3  # Cố định 3 đáp án sai / câu (format MCQ chuẩn)

# ══════════════════════════════════════════════════════════════
# SIDEBAR – Độ khó & Lịch sử
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Cấu hình")
    
    # ── Độ khó distractor ──────────────────────────────────────
    # Ảnh hưởng đến prompt LLM: Dễ = rõ ràng, Khó = gian lận cao
    difficulty = st.select_slider(
        "Độ khó đáp án sai",
        options=["Dễ", "Trung bình", "Khó"],
        value="Trung bình",
        help="Ảnh hưởng đến độ nhiễu trong prompt LLM",
    )
    
    st.divider()
    
    # ── Lịch sử ────────────────────────────────────────────────
    # Hiển thị 10 đề gần nhất, click để load lại
    st.markdown("## Lịch sử đề thi")
    
    hist = st.session_state["history"]
    if hist:
        st.markdown(f"**{len(hist)} đề gần nhất**")
        for i, h in enumerate(hist):
            # Mỗi đề: button với thông tin số câu
            if st.button(f"Đề #{i+1}  –  {len(h)} câu", key=f"hist_{i}"):
                # Load lại đề này vào mcq_list
                st.session_state["mcq_list"] = copy.deepcopy(h)
                st.session_state["editor_version"] += 1
                st.rerun()
    else:
        st.info("Chưa có lịch sử.\n\nSinh câu hỏi để lưu vào đây.")

    st.divider()
    st.caption("Sử dụng : shnl/vit5-vinewsqa-qg-ae + Ollama")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(
    "# Hệ thống tạo câu hỏi trắc nghiệm"
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
    st.markdown("#### Bước 1 – Nhập đoạn văn bản")

    col_ta, col_ex = st.columns([6, 1])
    with col_ex:
        st.markdown(" ")   # padding
        if st.button("\nVí dụ", use_container_width=True, help="Văn mẫu sẵn có"):
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
    st.markdown("#### Bước 2 – Sinh câu hỏi")
    num_pairs = st.number_input(
        "Số câu hỏi muốn sinh",
        min_value=1,
        max_value=20,
        value=num_pairs,
        step=1,
        help="Số cặp Q-A tối đa sẽ được sinh ra từ đoạn văn",
    )
    generate_btn = st.button(
        "Sinh câu hỏi trắc nghiệm",
        type="primary",
        use_container_width=True,
    )

    # ══════════════════════════════════════════════════════════
    # PIPELINE
    # ══════════════════════════════════════════════════════════
    if generate_btn:
        if not context or len(context.strip()) < 20:
            st.warning(" Đoạn văn quá ngắn. Nhập ít nhất 20 ký tự.")
        else:
            ctx = context.strip()
            st.session_state["context_buf"] = ctx
            mcq_list_new = []

            # difficulty → prompt modifier
            diff_hint = {
                "Dễ": "gần giống nhau về âm thanh hoặc cấu trúc",
                "Trung bình": "hợp lý và có sức nhiễu khá",
                "Khó": "rất dễ gây nhầm lẫn, tương tự đáp án đúng về ngữ nghĩa",
            }[difficulty]

            # ── Load models ───────────────────────────────────
            with st.status(" Load model…", expanded=True) as s:
                try:
                    st.write("Khởi động ViT5 (sinh câu hỏi)...")
                    qa_gen = load_qa_generator(viqag_model)
                    st.write("Đang kết nối đến Ollama...")
                    dist_gen = load_distractor_generator(ollama_host)
                    s.update(label="Kết nối thành công!", state="complete")
                except Exception as e:
                    s.update(label="Lỗi kết nối", state="error")
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
                        s.update(label="ViT5 không trả về kết quả", state="error")
                        st.warning(
                            "Không tìm thấy câu hỏi phù hợp.  \n"
                            "Thử: văn bản dài hơn hoặc chi tiết hơn."
                        )
                        st.stop()
                    s.update(
                        label=f"Stage 1 hoàn tất – {len(qa_pairs)} cặp Q-A",
                        state="complete",
                    )
                except Exception as e:
                    s.update(label="Lỗi ViT5", state="error")
                    st.error(f"ViT5 lỗi: {e}")
                    st.stop()

            # ── Quality filter Q-A pairs ───────────────────
            # Tính Jaccard similarity giữa 2 câu hỏi để phát hiện duplicate 
            def _jaccard(q1: str, q2: str) -> float:
                """Jaccard similarity = |giao| / |hợp| của tập từ.
                Số từ chung chia cho Tổng số từ
                Trả về 0.0-1.0, càng cao = càng giống nhau.
                """
                w1 = set(q1.lower().split())
                w2 = set(q2.lower().split())
                if not w1 or not w2:
                    return 0.0
                return len(w1 & w2) / len(w1 | w2)

            def _filter_qa(pairs):
                """Lọc chất lượng cặp Q-A từ ViT5.
                
                Loại bỏ:
                1. Câu trống hoặc answer trống
                2. Answer quá ngắn (<1 từ) hoặc quá dài (>30 từ)
                3. Answer leakage: answer nằm trong question
                4. Duplicate: câu hỏi giống nhau >= 40% (Jaccard)
                
                Returns:
                    List các cặp Q-A đạt chuẩn
                """
                seen_q = []  # Danh sách question đã thấy
                result = []
                for p in pairs:
                    
                    q_ = p.get("question", "").strip()
                    a_ = p.get("answer", "").strip()

                    if not q_ or not a_:
                        continue
                    # Bỏ answer quá dài (>30 từ)
                    if len(a_.split()) > 30:
                        continue
                    # Bỏ answer leakage: đáp án xuất hiện nguyên văn trong câu hỏi
                    if a_.lower() in q_.lower():
                        continue
                    # Bỏ câu hỏi trùng lặp: Jaccard similarity >= 40%
                    if any(_jaccard(q_, sq) >= 0.40 for sq in seen_q):
                        continue
                    seen_q.append(q_)
                    result.append(p)
                return result

            qa_pairs = _filter_qa(qa_pairs)[:num_pairs]  # giữ tối đa num_pairs sau lọc
            if not qa_pairs:
                st.warning("Sau khi lọc chất lượng, không còn câu hỏi phù hợp. Thử đoạn văn khác.")
                st.stop()

            # ── Stage 2: Ollama distractors ─────────────────────
            # Với mỗi cặp Q-A từ ViT5, gọi Ollama LLM sinh 3 đáp án sai
            progress_bar = st.progress(0, text="Stage 2 · Ollama – Đang sinh distractors…")
            errors_list  = []  # Track các câu lỗi

            for i, pair in enumerate(qa_pairs):
                q, a = pair["question"], pair["answer"]
                progress_bar.progress(
                    i / len(qa_pairs),
                    text=f"Stage 2 · Ollama – Câu {i+1}/{len(qa_pairs)}: {q[:45]}…",
                )
                # Throttle 0.5s: tránh overload Ollama khi sinh nhiều câu liên tiếp
                if i > 0:
                    time.sleep(0.5)
                try:
                    # Thêm hint độ khó vào context để ảnh hưởng LLM prompt
                    augmented_ctx = f"{ctx}\n[Yêu cầu: distractors {diff_hint}]"
                    distractors = dist_gen.generate(
                        question=q, answer=a,
                        context=augmented_ctx,
                        num_distractors=num_distractors,
                    )
                    # Nếu LLM trả về < 3 distractors → thêm placeholder
                    while len(distractors) < num_distractors:
                        distractors.append(f"[Đáp án sai {len(distractors)+1}]")
                    mcq_list_new.append(build_mcq(q, a, distractors[:num_distractors]))
                except Exception as e:
                    print(f"[Distractor] FAILED câu {i+1}: {e}")
                    errors_list.append(f"Câu {i+1}: {e}")
                    # Fallback: vẫn thêm câu với placeholder distractor (user sửa sau)
                    placeholders = [f"[Đáp án sai {j+1}]" for j in range(num_distractors)]
                    mcq_list_new.append(build_mcq(q, a, placeholders))

            progress_bar.progress(1.0, text="Stage 2 hoàn tất!")
            time.sleep(0.3)
            progress_bar.empty()

            # Hiển thị cảnh báo nếu có câu lỗi
            if errors_list:
                st.warning(
                    f"LLM distractor gặp lỗi {len(errors_list)}/{len(qa_pairs)} câu "
                    f"(dùng placeholder tạm - user có thể edit hoặc regenerate sau). "
                    f"Lỗi: {str(errors_list[0])[:120]}"
                )

            # Lưu vào session state và history
            st.session_state["mcq_list"] = mcq_list_new
            st.session_state["selected"] = set(range(len(mcq_list_new)))  # Chọn tất cả mặc định
            st.session_state["editor_version"] += 1
            save_to_history(mcq_list_new)
            if mcq_list_new:
                st.success(f"🎉 Hoàn tất! Đã tạo **{len(mcq_list_new)} câu trắc nghiệm**.")
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # DANH SÁCH CÂU HỎI – chỉnh sửa / xóa / regenerate
    # ══════════════════════════════════════════════════════════
    # Sau khi generate, hiển thị danh sách câu hỏi cho user xem/edit
    mcq_list: List[Dict] = st.session_state["mcq_list"]

    if mcq_list:
        st.divider()
        st.markdown(f"#### Bước 3 – Xem và chỉnh sửa ({len(mcq_list)} câu)")
        editor_version = st.session_state["editor_version"]

        # Thanh hành động nhanh: batch operations cho nhiều câu
        act1, act2, act3 = st.columns(3)
        with act1:
            # Chọn tất cả câu để export
            if st.button("Chọn tất cả"):
                st.session_state["selected"] = set(range(len(mcq_list)))
                st.session_state["editor_version"] += 1
                st.rerun()
        with act2:
            # Bỏ chọn tất cả
            if st.button("Bỏ chọn tất cả"):
                st.session_state["selected"] = set()
                st.session_state["editor_version"] += 1
                st.rerun()
        with act3:
            # Thêm câu trống (user tự điền)
            if st.button("Thêm câu trống"):
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

        # ── Render từng card câu hỏi ─────────────────────────────
        # Mỗi câu hiển thị: checkbox, nội dung, edit, actions (shuffle/regen/delete)
        to_delete = None  # Track index câu cần xóa (xử lý sau vòng lặp)

        for idx, mcq in enumerate(mcq_list):
            is_selected = idx in st.session_state["selected"]
            card_cls    = "mcq-card selected" if is_selected else "mcq-card"

            with st.container():
                # Header card: checkbox + số câu + nguồn (ViT5+LLM / thủ công)
                hc1, hc2 = st.columns([1, 11])
                with hc1:
                    # Checkbox để chọn câu này export
                    checked = st.checkbox(
                        f"Chọn câu {idx+1}", value=is_selected,
                        key=f"sel_{editor_version}_{idx}",
                        label_visibility="collapsed",
                    )
                    # Cập nhật session state khi checkbox thay đổi
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

                # Nội dung câu hỏi (có thể chỉnh sửa)
                with st.expander(f"{mcq['question'][:80]}{'…' if len(mcq['question'])>80 else ''}", expanded=False):
                    # Edit question text
                    new_q = st.text_input(
                        "Nội dung câu hỏi",
                        value=mcq["question"],
                        key=f"q_{editor_version}_{idx}",
                    )
                    if new_q != mcq["question"]:
                        st.session_state["mcq_list"][idx]["question"] = new_q
                        st.rerun()

                    st.markdown("**Đáp án:**")
                    # Render 4 options: A, B, C, D
                    for j, opt in enumerate(mcq["options"]):
                        lbl        = LABELS[j]
                        is_correct = (lbl == mcq["correct_label"])
                        c1, c2, c3 = st.columns([1, 8, 3])
                        with c1:
                            st.markdown(
                                f'<span style="color:{"#16a34a" if is_correct else "#6b7280"};'
                                f'font-weight:{"700" if is_correct else "400"}">{lbl}.</span>',
                                unsafe_allow_html=True,
                            )
                        with c2:
                            # Edit option text
                            new_opt = st.text_input(
                                f"opt_{idx}_{j}",
                                value=opt,
                                key=f"opt_{editor_version}_{idx}_{j}",
                                label_visibility="collapsed",
                            )
                            if new_opt != opt:
                                # Cập nhật option mới
                                st.session_state["mcq_list"][idx]["options"][j] = new_opt
                                # Nếu option này là đáp án đúng, cập nhật answer
                                if is_correct:
                                    st.session_state["mcq_list"][idx]["answer"] = new_opt
                                st.rerun()
                        with c3:
                            # Nút "set làm đáp án đúng" (chỉ hiển với đáp án sai)
                            if not is_correct:
                                if st.button("Đặt đúng", key=f"setcorrect_{editor_version}_{idx}_{j}"):
                                    st.session_state["mcq_list"][idx]["correct_label"] = lbl
                                    st.session_state["mcq_list"][idx]["answer"] = opt
                                    st.rerun()

                    # Nút hành động câu: regenerate / delete
                    spacer, rb1, rb2 = st.columns([6, 3, 2])
                    with spacer:
                        st.empty() # Không vẽ gì vào đây cả, chỉ để đẩy 2 nút kia sang phải
                    with rb1:
                        # Regenerate distractors bằng Ollama (giữ nguyên question/answer)
                        if st.button("Tạo lại đáp án", key=f"regen_{editor_version}_{idx}"):
                            st.session_state["regen_idx"] = idx
                            st.rerun()
                    with rb2:
                        # Xóa câu này
                        if st.button("Xóa câu", key=f"del_{editor_version}_{idx}", type="secondary"):
                            to_delete = idx

        # Xóa câu (sau vòng lặp để tránh lỗi index)
        if to_delete is not None:
            st.session_state["mcq_list"].pop(to_delete)
            st.session_state["selected"].discard(to_delete)  # Bỏ khỏi selected set
            # Re-index selected: các câu sau to_delete phải giảm index xuống 1
            st.session_state["selected"] = {
                i if i < to_delete else i - 1
                for i in st.session_state["selected"] if i != to_delete
            }
            st.rerun()

        # ── Regenerate single question ────────────────────────
        regen_idx = st.session_state.get("regen_idx")
        if regen_idx is not None and 0 <= regen_idx < len(mcq_list):
            st.session_state["regen_idx"] = None
            with st.spinner(f"Đang regenerate distractors câu {regen_idx+1}…"):
                try:
                    dist_gen = load_distractor_generator(ollama_host)
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
                    st.session_state["editor_version"] += 1
                    st.success(f"Đã cập nhật câu {regen_idx+1}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi regenerate: {e}")

    else:
        # Màn chào ban đầu (khi chưa có câu hỏi nào)
        st.info(
            "**Hướng dẫn nhanh:**\n"
            "1. Dán đoạn văn bản vào ô phía trên\n"
            "2. Bấm **Sinh câu hỏi trắc nghiệm**\n"
            "3. Chỉnh sửa câu hỏi nếu cần\n"
            "4. Export Word/PDF từ cột bên phải"
        )
        with st.expander("Về pipeline ViT5 + Ollama"):
            st.markdown("""
| Thành phần | Vai trò |
|---|---|
| **ViT5 (Local)** | Model `shnl/vit5-vinewsqa-qg-ae` sinh câu hỏi & trích đáp án từ văn bản tiếng Việt |
| **Ollama (Local)** | Sinh 3 đáp án sai (llama3 / qwen2 / gemma) |
| **MCQ Builder** | Xáo trộn đáp án, đánh nhãn A/B/C/D |
            """)


# ──────────────────────────────────────────────────────────────
# CỘT PHẢI – Preview & Export
# ──────────────────────────────────────────────────────────────
# Hiển thị preview text của các câu được chọn + nút tải về định dạng
with col_preview:
    mcq_list: List[Dict] = st.session_state["mcq_list"]
    selected: set        = st.session_state["selected"]
    # Chỉ export các câu được chọn (checked)
    export_list          = [mcq_list[i] for i in sorted(selected) if i < len(mcq_list)]

    st.markdown("#### Xem trước & Tải về")

    if export_list:
        st.caption(f"**{len(export_list)}/{len(mcq_list)}** câu được chọn để xuất")

        # Preview text (không có đáp án)
        preview_txt = mcq_to_text(export_list, show_ans=False)
        st.markdown(
            f'<div class="preview-box">{preview_txt}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Tải về")

        # ── JSON ───────────────────────────────────────────────
        # Format chuẩn dùng import vào Quizizz, Google Forms, hoặc lưu trữ
        st.download_button(
            "JSON (import Quizizz / Google Forms)",
            data=json.dumps(export_list, ensure_ascii=False, indent=2),
            file_name="mcq.json",
            mime="application/json",
            use_container_width=True,
        )

        # ── TXT ──────────────────────────────────────────────────
        # Text thuần có đáp án, dễ đọc/in ấn
        st.download_button(
            "TXT (kèm đáp án)",
            data=mcq_to_text(export_list, show_ans=True),
            file_name="mcq.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ── Word ───────────────────────────────────────────────
        # Xuất .docx: câu hỏi trang trước, đáp án trang sau
        try:
            from export_utils import export_word_bytes
            word_bytes = export_word_bytes(export_list)
            st.download_button(
                "Word (.docx) – có đáp án trang sau",
                data=word_bytes,
                file_name="mcq.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Word: {e}")

        # ── PDF ────────────────────────────────────────────────
        # Xuất PDF: câu hỏi trang trước, đáp án trang sau
        try:
            from export_utils import export_pdf_bytes
            pdf_bytes = export_pdf_bytes(export_list)
            st.download_button(
                "PDF – có đáp án trang sau",
                data=pdf_bytes,
                file_name="mcq.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF: {e}")

        st.divider()

        # ── Bảng đáp án ───────────────────────────────────────
        # Hiển thị bảng markdown: Câu | Đáp án
        st.markdown("##### Bảng đáp án")
        ans_md = "| Câu | Đáp án |\n|---|---|\n"
        for i, m in enumerate(export_list, 1):
            ans_md += f"| {i} | **{m['correct_label']}**. {m['answer']} |\n"
        st.markdown(ans_md)

    else:
        # Khi chưa chọn câu nào: hướng dẫn user tick checkbox
        st.info("Chọn câu hỏi ở cột trái (checkbox ☑) để xem preview và xuất đề.")

        if mcq_list:
            st.markdown(f"Hiện có **{len(mcq_list)} câu** chưa được chọn.")
            # Quick action: chọn tất cả luôn
            if st.button("Chọn tất cả & preview"):
                st.session_state["selected"] = set(range(len(mcq_list)))
                st.rerun()




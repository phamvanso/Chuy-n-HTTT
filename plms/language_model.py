# ============================================================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ============================================================================
import os
import logging
import pickle  # Dùng để lưu/đọc object Python dưới dạng binary
import re  # Regular expression để xử lý chuỗi
import urllib  # Kiểm tra kết nối internet
from itertools import chain  # Nối nhiều list lại thành một
from typing import List, Dict  # Type hints cho Python
from multiprocessing import Pool  # Xử lý song song đa luồng
import numpy as np
from tqdm import tqdm  # Thanh tiến trình
import torch  # PyTorch framework
from torch.nn import functional  # Hàm loss và activation
import transformers  # Thư viện Hugging Face Transformers
from .exceptions import ExceedMaxLengthError, HighlightNotFoundError, AnswerNotFoundError
from .spacy_module import SpacyPipeline, VALID_METHODS

# Xuất các class/function chính của module này
__all__ = ('TransformersQG', 'ADDITIONAL_SP_TOKENS', 'TASK_PREFIX', 'clean', 'internet_connection')

# ============================================================================
# THIẾT LẬP CẤU HÌNH VÀ HẰNG SỐ
# ============================================================================
# Tắt cảnh báo parallelism của tokenizer để log gọn hơn
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prefix theo dõi nhiệm vụ: cho model biết đang thực hiện task nào
# ae = Answer Extraction (trích xuất câu trả lời)
# qg = Question Generation (sinh câu hỏi)
# qag = Question-Answer Generation (sinh câu hỏi và đáp án)
# qa = Question Answering (trả lời câu hỏi)
TASK_PREFIX = {
    "ae": "extract answers",
    "qg": "generate question",
    "qag": "generate question and answer",
    "qa": "answer question"
}

# Index dùng để bỏ qua padding token trong loss function
CE_IGNORE_INDEX = -100

# Token đặc biệt dùng để highlight (đánh dấu) câu trả lời trong văn bản
# Ví dụ: "Hà Nội là <hl> thủ đô <hl> của Việt Nam"
ADDITIONAL_SP_TOKENS = {'hl': '<hl>'}

# Số worker cho DataLoader (đọc từ biến môi trường)
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '0'))

# Có sử dụng xử lý song song (multiprocessing) hay không
PARALLEL_PROCESSING = bool(int(os.getenv('PARALLEL_PROCESSING', '0')))

# Model mặc định cho từng ngôn ngữ
DEFAULT_MODELS = {
    'vi': 'VietAI/vit5-base'  # Model tiếng Việt từ VietAI
}


# ============================================================================
# HÀM TIỆN ÍCH (UTILITY FUNCTIONS)
# ============================================================================

def pickle_save(obj, path: str):
    """Lưu object Python ra file binary bằng pickle.
    
    Args:
        obj: Object Python cần lưu (có thể là dict, list, model state, v.v.)
        path: Đường dẫn file đích để lưu
    """
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    """Đọc lại object Python từ file pickle đã lưu trước đó.
    
    Args:
        path: Đường dẫn file pickle cần đọc
        
    Returns:
        Object Python đã được lưu trong file
    """
    with open(path, "rb") as fp:
        return pickle.load(fp)


def clean(string):
    """Xóa khoảng trắng thừa ở đầu và cuối chuỗi.
    
    Args:
        string: Chuỗi cần làm sạch
        
    Returns:
        Chuỗi đã xóa khoảng trắng, hoặc None nếu chuỗi rỗng
    """
    # Xóa khoảng trắng ở đầu chuỗi
    string = re.sub(r'\A\s*', '', string)
    # Xóa khoảng trắng ở cuối chuỗi
    string = re.sub(r'\s*\Z', '', string)
    if len(string) > 0:
        return string
    return None


def internet_connection(host='http://google.com'):
    """Kiểm tra xem có kết nối internet hay không.
    
    Args:
        host: URL để thử kết nối (mặc định là google.com)
        
    Returns:
        True nếu có internet, False nếu không có
    """
    try:
        urllib.request.urlopen(host)
        return True  # Kết nối thành công
    except:
        return False  # Không có kết nối hoặc lỗi


def load_language_model(model_name,
                        cache_dir: str = None,
                        use_auth_token: bool = False,
                        torch_dtype=None,
                        device_map: str = None,
                        low_cpu_mem_usage: bool = False):
    """Tải tokenizer và model từ Hugging Face, bổ sung special token cần thiết.

    Hàm này thực hiện các bước:
    1. Tự động chọn local_files_only=True khi không có internet
    2. Tải tokenizer và config từ model_name
    3. Chọn đúng class model theo config.model_type (T5, MT5, BART, MBART, etc.)
    4. Thêm token đặc biệt <hl> để đánh dấu span highlight trong input
    5. Resize embedding layer để phù hợp với tokenizer mới
    
    Args:
        model_name: Tên model trên Hugging Face hub hoặc đường dẫn local
        cache_dir: Thư mục cache để lưu model/tokenizer
        use_auth_token: Token xác thực Hugging Face (cho private model)
        torch_dtype: Kiểu dữ liệu tensor (float32, float16, bfloat16, etc.)
        device_map: Sơ đồ phân bổ model lên các thiết bị
        low_cpu_mem_usage: Giảm bộ nhớ CPU khi tải model
        
    Returns:
        tuple: (tokenizer, model, config)
    """
    # Kiểm tra kết nối internet: nếu không có thì chỉ dùng file local
    local_files_only = not internet_connection()
    
    # Tải tokenizer (bộ mã hóa văn bản thành số)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only, use_auth_token=use_auth_token)
    
    # Tải config (cấu hình model: số layer, hidden size, v.v.)
    config = transformers.AutoConfig.from_pretrained(
        model_name, local_files_only=local_files_only, cache_dir=cache_dir, use_auth_token=use_auth_token)
    
    # Chọn class model phù hợp theo loại kiến trúc (mỗi loại có class riêng)
    if config.model_type == 't5':  # T5 model requires T5ForConditionalGeneration class
        model_class = transformers.T5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'mt5':
        model_class = transformers.MT5ForConditionalGeneration.from_pretrained
    elif config.model_type == 'bart':
        model_class = transformers.BartForConditionalGeneration.from_pretrained
    elif config.model_type == 'mbart':
        model_class = transformers.MBartForConditionalGeneration.from_pretrained
    elif config.model_type == 'switch_transformers':
        model_class = transformers.SwitchTransformersForConditionalGeneration.from_pretrained
    else:
        raise ValueError(f'unsupported model type: {config.model_type}')

    # Chuẩn bị các tham số để tải model
    param = {'config': config, "local_files_only": local_files_only, "use_auth_token": use_auth_token,
             "low_cpu_mem_usage": low_cpu_mem_usage, "cache_dir": cache_dir}
    
    # Thêm torch_dtype nếu được chỉ định (ví dụ: float16 để tiết kiệm bộ nhớ)
    if torch_dtype is not None:
        param['torch_dtype'] = torch_dtype
    
    # Thêm device_map nếu được chỉ định (phân bổ model lên nhiều GPU)
    if device_map is not None:
        param['device_map'] = device_map
    
    # Tải model với các tham số đã chuẩn bị
    model = model_class(model_name, **param)
    
    # Đồng bộ tokenizer/model với token đặc biệt (<hl>) dùng trong pipeline QG
    # Token <hl> dùng để đánh dấu câu trả lời trong context
    tokenizer.add_special_tokens({'additional_special_tokens': list(ADDITIONAL_SP_TOKENS.values())})
    
    # Resize embedding layer của model để khớp với số token mới trong tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model, config


def label_smoothed_loss(logits, labels, epsilon):
    """Tính label smoothing loss để regularization, tránh overfitting.
    
    Label smoothing: thay vì target là 1.0 cho class đúng và 0.0 cho class khác,
    ta dùng (1-epsilon) cho class đúng và epsilon/(num_classes-1) cho các class khác.
    Điều này giúp model không quá tự tin vào prediction.
    
    Args:
        logits: Output logits từ model (chưa qua softmax)
        labels: Ground truth labels
        epsilon: Hệ số smoothing (thường là 0.1)
        
    Returns:
        Loss value đã được smoothed
        
    Reference:
        https://github.com/huggingface/transformers/blob/55bb4c06f7be141c6d895dbe1f11018dc8580b2d/src/transformers/trainer_pt_utils.py#L430
    """
    # Tính log probability (negative log softmax)
    log_probs = - functional.log_softmax(logits, dim=-1)
    
    # Thêm chiều cuối nếu labels thiếu chiều (để match với log_probs)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    # Tạo mask để đánh dấu các vị trí padding (ignore index = -100)
    padding_mask = labels.eq(CE_IGNORE_INDEX)
    
    # Nếu label = -100 thì gather sẽ lỗi, nên tạm thời clamp về 0
    # Các vị trí pad vẫn được loại bỏ bởi padding_mask nên không ảnh hưởng kết quả
    labels.clamp_min_(0)

    # Tính negative log-likelihood loss (cross-entropy loss thông thường)
    # Gather lấy ra log prob tại vị trí của label đúng
    nll_loss = log_probs.gather(dim=-1, index=labels)
    nll_loss.masked_fill_(padding_mask, 0.0)  # Bỏ qua các vị trí padding

    # Tính smoothed loss: trung bình của tất cả log probs
    # Công thức này vẫn ổn với fp16 vì tổng được tính ở fp32 (tránh overflow)
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
    smoothed_loss.masked_fill_(padding_mask, 0.0)  # Bỏ qua các vị trí padding

    # Chuẩn hóa theo số token hợp lệ (không tính padding)
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    
    # Kết hợp NLL loss và smoothed loss theo hệ số epsilon
    # (1-epsilon) * loss_chuan + epsilon * loss_smoothed
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


# ============================================================================
# CLASS DATASET VÀ ENCODING
# ============================================================================

class Dataset(torch.utils.data.Dataset):
    """Wrapper nhỏ cho DataLoader: chuyển dict feature thành tensor.
    
    Class này kế thừa torch.utils.data.Dataset để có thể dùng với DataLoader.
    Nhiệm vụ chính: chuyển đổi dictionary features thành tensor PyTorch.
    """
    # Các field cần chuyển thành float tensor (còn lại là long tensor)
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        """Khởi tạo dataset với list các data sample.
        
        Args:
            data: List các dict, mỗi dict chứa feature của 1 sample
        """
        self.data = data

    def __len__(self):
        """Trả về số lượng sample trong dataset."""
        return len(self.data)

    def to_tensor(self, name, data):
        """Chuyển data thành tensor với dtype phù hợp.
        
        Args:
            name: Tên field (để xác định dtype)
            data: Dữ liệu cần chuyển
            
        Returns:
            Tensor với dtype phù hợp
        """
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)  # Mặc định là long (cho input_ids, labels)

    def __getitem__(self, idx):
        """Lấy sample thứ idx và chuyển tất cả field thành tensor.
        
        Args:
            idx: Chỉ số sample cần lấy
            
        Returns:
            Dict với key giống input nhưng value đã là tensor
        """
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class EncodePlus:
    """Wrapper cho bước tokenize input/output, có thể dùng trong multiprocessing.
    
    Class này đóng gói logic tokenize để:
    1. Có thể serialize và dùng trong Pool.map() (multiprocessing)
    2. Tập trung xử lý highlight span (<hl>)
    3. Xử lý các lỗi: overflow, highlight không tìm thấy
    4. Thêm task prefix nếu cần (qg:, ae:, qa:, qag:)
    """

    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 max_length_output: int = 34,
                 drop_overflow_error_text: bool = False,
                 skip_overflow_error: bool = False,
                 drop_highlight_error_text: bool = False,
                 prefix_type: str = None,
                 padding: bool = True):
        """Khởi tạo bộ mã hóa input/output với các tùy chọn xử lý lỗi.

        Args:
            tokenizer: Tokenizer từ transformers
            max_length: Độ dài tối đa của input sequence (token)
            max_length_output: Độ dài tối đa của output sequence (token)
            drop_overflow_error_text: True -> bỏ mẫu vượt quá max length (trả về None)
                                      False -> raise exception khi gặp mẫu quá dài
            skip_overflow_error: True -> bỏ qua check overflow (không kiểm tra độ dài)
            drop_highlight_error_text: True -> bỏ mẫu nếu không tìm thấy highlight span
                                       False -> raise exception khi không tìm thấy
            prefix_type: Prefix nhiệm vụ ('qg', 'ae', 'qag', 'qa') để thêm vào đầu input
            padding: Có pad về max_length hay không (True cho training, False cho inference)
        """
        # Lấy text prefix theo task (ví dụ: "generate question:")
        self.prefix = TASK_PREFIX[prefix_type] if prefix_type is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_output = max_length_output
        
        # Khi train thường bỏ mẫu overlength; khi evaluate có thể muốn giữ lại
        self.drop_overflow_error_text = drop_overflow_error_text
        self.skip_overflow_error = skip_overflow_error
        self.drop_highlight_error_text = drop_highlight_error_text
        
        # Bật truncation cho pipeline batch để tránh vỡ batch (truncate thay vì raise error)
        self.param_in = {'truncation': True, 'max_length': self.max_length}
        self.param_out = {'truncation': True, 'max_length': self.max_length_output}
        
        # Thêm padding nếu cần (thường dùng cho training để tất cả sample cùng độ dài)
        if padding:
            self.param_in['padding'] = 'max_length'
            self.param_out['padding'] = 'max_length'

    def __call__(self, inputs):
        return self.encode_plus(*inputs)

    def encode_plus(self, input_sequence: str, output_sequence: str = None, input_highlight: str = None):
        """Tokenize một mẫu dữ liệu thành input_ids và labels.

        Luồng xử lý:
        1. Nếu có input_highlight: chèn token <hl> quanh span đó trong input
        2. Thêm task prefix vào đầu input nếu cần
        3. Kiểm tra độ dài input/output, drop hoặc raise error nếu quá dài
        4. Tokenize input và output thành numbers
        
        Args:
            input_sequence: Câu/văn bản đầu vào (context)
            output_sequence: Chuỗi mục tiêu (nếu có, dùng cho training)
            input_highlight: Chuỗi con cần đánh dấu bằng <hl> (thường là answer)
            
        Returns:
            Dict feature đã tokenized (input_ids, attention_mask, có thể kèm labels)
            hoặc None nếu drop_overflow_error_text=True và mẫu quá dài
        """
        # Bước 1: Chèn token <hl> quanh span câu trả lời để model biết vị trí cần hỏi
        # Ví dụ: "Hà Nội là thủ đô" + highlight="thủ đô" -> "Hà Nội là <hl> thủ đô <hl>"
        if input_highlight is not None:
            position = input_sequence.find(input_highlight)
            if position == -1:
                if self.drop_highlight_error_text:
                    return None
                raise HighlightNotFoundError(input_highlight, input_sequence)
            input_sequence = '{0}{1} {2} {1}{3}'.format(
                input_sequence[:position], ADDITIONAL_SP_TOKENS['hl'], input_highlight,
                input_sequence[position+len(input_highlight):])
        if self.prefix is not None:
            input_sequence = f'{self.prefix}: {input_sequence}'

        # Bước 3: Xử lý overlength (mẫu quá dài)
        # Có 3 chế độ:
        # - drop_overflow_error_text=True: loại bỏ mẫu (return None)
        # - skip_overflow_error=True: bỏ qua check (để truncate tự động xử lý)
        # - Mặc định: raise exception để phát hiện dữ liệu lỗi
        if self.drop_overflow_error_text or not self.skip_overflow_error:
            # Kiểm tra độ dài input sequence
            if len(self.tokenizer.encode(input_sequence)) > self.max_length:
                if not self.drop_overflow_error_text:  # Báo lỗi nếu không cho phép drop
                    raise ExceedMaxLengthError(self.max_length)
                return None  # Loại bỏ mẫu overlength
            
            # Kiểm tra độ dài output sequence (nếu có)
            if output_sequence is not None:
                if len(self.tokenizer.encode(output_sequence)) > self.max_length_output:
                    if not self.drop_overflow_error_text:  # Báo lỗi nếu output quá dài
                        raise ExceedMaxLengthError(self.max_length)
                    return None  # Loại bỏ mẫu overlength
        if type(self.tokenizer) is transformers.models.mbart.tokenization_mbart_fast.MBartTokenizerFast:
            encode = self.tokenizer(input_sequence, **self.param_in)
        else:
            encode = self.tokenizer(text_target=input_sequence, **self.param_in)
        if output_sequence is not None:
            encode['labels'] = self.tokenizer.encode(output_sequence, **self.param_out)
        return encode


# ============================================================================
# CLASS TRANSFORMERSQG - MODEL CHÍNH
# ============================================================================

class TransformersQG:
    """Model wrapper cho các tác vụ QG/AE/QA/QAG trên Transformers.
    
    Class chính của module, cung cấp interface thống nhất cho 4 tác vụ:
    - QG (Question Generation): Sinh câu hỏi từ context + answer
    - AE (Answer Extraction): Trích xuất câu trả lời từ context
    - QA (Question Answering): Trả lời câu hỏi từ context
    - QAG (Question-Answer Generation): Sinh cặp (question, answer) từ context
    
    Có thể dùng:
    - Multitask model: 1 model làm nhiều task
    - Pipeline model: model AE riêng + model QG riêng
    - SpaCy backend: dùng spaCy cho AE (keyword extraction)
    """

    def __init__(self,
                 model: str = None,
                 max_length: int = 512,
                 max_length_output: int = 256,
                 model_ae: str = None,
                 max_length_ae: int = 512,
                 max_length_output_ae: int = 64,
                 cache_dir: str = None,
                 add_prefix: bool = None,
                 language: str = 'vi',
                 label_smoothing: float = None,
                 skip_overflow_error: bool = False,
                 drop_overflow_error_text: bool = False,
                 drop_highlight_error_text: bool = False,
                 drop_answer_error_text: bool = False,
                 use_auth_token: bool = False,
                 torch_dtype=None,
                 device_map: str = None,
                 low_cpu_mem_usage: bool = False,
                 is_qg: bool = None,
                 is_qag: bool = None,
                 is_qa: bool = None,
                 is_ae: bool = None):
        """Khởi tạo model và các thành phần phụ trợ cho sinh câu hỏi.

        Args:
            model: Tên model trên Hugging Face hub hoặc đường dẫn local
            max_length: Độ dài tối đa input (số token)
            max_length_output: Độ dài tối đa output (số token)
            model_ae: Tên model cho Answer Extraction (nếu riêng)
            max_length_ae: Độ dài tối đa input cho model AE
            max_length_output_ae: Độ dài tối đa output cho model AE
            cache_dir: Thư mục cache model/tokenizer
            add_prefix: Có thêm prefix tác vụ vào input hay không
            language: Ngôn ngữ dùng cho pipeline spaCy (ví dụ: 'vi', 'en')
            label_smoothing: Hệ số label smoothing khi train (0.0-1.0, thường 0.1)
            drop_overflow_error_text: True -> bỏ mẫu overlength thay vì raise error
            skip_overflow_error: True -> bỏ qua check overlength
            drop_highlight_error_text: True -> bỏ mẫu nếu không tìm thấy highlight
            drop_answer_error_text: True -> bỏ mẫu nếu không tìm thấy answer
            use_auth_token: Token Hugging Face cho private model
            torch_dtype: Kiểu dữ liệu tensor (float32, float16, bfloat16)
            device_map: Sơ đồ phân bổ model lên GPU
            low_cpu_mem_usage: Giảm bộ nhớ CPU khi tải model
            is_qg: Model có hỗ trợ Question Generation không
            is_qag: Model có hỗ trợ Question-Answer Generation không
            is_qa: Model có hỗ trợ Question Answering không
            is_ae: Model có hỗ trợ Answer Extraction không
        """

        # Bước 1: Nếu không truyền model, lấy model mặc định theo ngôn ngữ
        if model is None:
            assert language in DEFAULT_MODELS.keys(),\
                f"Model with language '{language}' is not available. Please choose language from " \
                f"'{DEFAULT_MODELS.keys()}' or specify 'model'."
            model = DEFAULT_MODELS[language]  # Ví dụ: 'vi' -> 'VietAI/vit5-base'

        # Bước 2: Suy ra khả năng tác vụ từ tên model (qg/ae/qa/qag)
        # Ví dụ: 'vit5-base-qg' -> is_qg=True
        self.is_qg = 'qg' in model.split('-') if is_qg is None else is_qg
        self.is_ae = 'ae' in model.split('-') if is_ae is None else is_ae
        self.is_qa = 'qa' in model.split('-') if is_qa is None else is_qa
        self.is_qag = 'qag' in model.split('-') if is_qag is None else is_qag
        
        # Bước 3: Lưu cấu hình cơ bản
        self.model_name = model
        self.max_length = max_length
        self.max_length_output = max_length_output
        self.label_smoothing = label_smoothing
        self.drop_overflow_error_text = drop_overflow_error_text
        self.skip_overflow_error = skip_overflow_error
        self.drop_highlight_error_text = drop_highlight_error_text
        self.drop_answer_error_text = drop_answer_error_text
        self.model_name_ae = model_ae
        self.max_length_ae = max_length_ae
        self.max_length_output_ae = max_length_output_ae
        # Bước 4: Nạp model chính (QG/QA/QAG) từ Hugging Face
        self.tokenizer, self.model, config = load_language_model(
            self.model_name, cache_dir=cache_dir, use_auth_token=use_auth_token, device_map=device_map,
            torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        
        # Kiểm tra xem model đã được fine-tune chưa (có add_prefix trong config không)
        if 'add_prefix' not in config.to_dict().keys():
            # Model chưa có flag add_prefix trong config (thường là base model chưa fine-tune)
            # assert add_prefix, '`add_prefix` is required for non-fine-tuned models'
            self.add_prefix = add_prefix
        else:
            # Model đã fine-tune, lấy config từ model
            self.add_prefix = config.add_prefix

        # Bước 5: Cài đặt mặc định cho module trích xuất đáp án (AE)
        # Có 3 lựa chọn:
        # 1. spaCy backend: dùng thuật toán NLP truyền thống (positionrank, textrank, v.v.)
        # 2. Multitask: cùng 1 model làm cả AE và QG
        # 3. Pipeline: model AE riêng + model QG riêng
        if self.model_name_ae is None:
            # Nếu không chỉ định model_ae:
            # - Nếu model chính hỗ trợ AE -> dùng model chính
            # - Không thì dùng positionrank (spaCy backend mặc định)
            self.model_name_ae = self.model_name if self.is_ae else "positionrank"
        
        # Chọn backend AE: spaCy / multitask (chung model) / pipeline (model riêng)
        self.answer_model_type = None
        
        # Kiểm tra xem model_ae có phải là phương pháp spaCy không
        if self.model_name_ae in VALID_METHODS:
            # Sử dụng spaCy backend (positionrank, textrank, yake, v.v.)
            logging.info(f'use spaCy answer extraction model: {self.model_name_ae}')
            self.tokenizer_ae = self.model_ae = self.add_prefix_ae = None
            self.spacy_module = SpacyPipeline(language, self.model_name_ae)
            self.answer_model_type = 'spacy'
        else:
            # Sử dụng LMQG fine-tuned model cho AE
            logging.info(f'use LMQG fine-tuned answer extraction model: {self.model_name_ae}')
            
            if self.model_name == self.model_name_ae:
                # Multitask: cùng 1 model làm cả AE và QG
                logging.info("the same model as QG is used as AE")
                assert self.is_ae, f"the model ({self.model_name_ae}) is not fine-tuned for AE"
                self.tokenizer_ae = self.model_ae = self.add_prefix_ae = None
                self.answer_model_type = 'multitask'
            else:
                # Pipeline: model AE riêng
                logging.info(f"loading 2nd model for AE: {self.model_name_ae}")
                self.tokenizer_ae, self.model_ae, config_ae = load_language_model(model_ae, cache_dir=cache_dir, use_auth_token=use_auth_token)
                self.add_prefix_ae = config_ae.add_prefix
                self.answer_model_type = 'pipeline'
            
            # Dù sao cũng cần spaCy để tách câu (sentence segmentation)
            self.spacy_module = SpacyPipeline(language)

        # Bước 6: Thiết lập thiết bị tính toán (CPU/GPU) và DataParallel nếu nhiều GPU
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False  # Flag đánh dấu có dùng DataParallel không
        
        # Nếu có nhiều hơn 1 GPU -> sử dụng DataParallel để phân tán tính toán
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)  # Wrap model chính
            if self.model_ae is not None:
                self.model_ae = torch.nn.DataParallel(self.model_ae)  # Wrap model AE nếu có
        
        # Chuyển model lên device (GPU hoặc CPU)
        self.model.to(self.device)
        if self.model_ae is not None:
            self.model_ae.to(self.device)
        
        # Log thông tin cấu hình
        logging.info(f'Model `{self.model_name}`')
        logging.info(f'\t * Num of GPU in use: {torch.cuda.device_count()}')
        logging.info(f'\t * Prefix: {self.add_prefix}')
        logging.info(f'\t * Language: {language} (ignore at the training phase)')

    def push_to_hub(self, repo_id):
        """Push model và tokenizer lên Hugging Face Hub.
        
        Args:
            repo_id: Tên repository trên Hugging Face (format: username/model_name)
        """
        # Nếu dùng DataParallel, phải unwrap model (lấy self.model.module)
        if self.parallel:
            self.model.module.push_to_hub(repo_id)
        else:
            self.model.push_to_hub(repo_id)
        self.tokenizer.push_to_hub(repo_id)

    def generate_qa_end2end(self,
                            list_context: str or List,
                            batch_size: int = None,
                            num_beams: int = 4,
                            cache_path: str = None,
                            splitting_symbol: str = ' [SEP] ',
                            question_prefix: str = "question: ",
                            answer_prefix: str = ", answer: "):
        """Sinh trực tiếp danh sách (question, answer) từ context bằng model QAG end-to-end.

        Model QAG end-to-end sinh ra chuỗi dạng:
        "question: Câu hỏi 1, answer: Câu trả lời 1 [SEP] question: Câu hỏi 2, answer: Câu trả lời 2"
        Hàm này parse chuỗi đó thành danh sách các cặp (question, answer).
        
        Args:
            list_context: Một context hoặc danh sách context
            batch_size: Batch size khi generate (tốc độ xử lý)
            num_beams: Số beam search (cao hơn = chất lượng tốt hơn nhưng chậm hơn)
            cache_path: Đường dẫn cache feature đã encode (tăng tốc độ)
            splitting_symbol: Ký hiệu phân tách giữa các cặp QA (mặc định: ' [SEP] ')
            question_prefix: Prefix trước question trong output (mặc định: "question: ")
            answer_prefix: Prefix trước answer trong output (mặc định: ", answer: ")
            
        Returns:
            Danh sách các cặp (question, answer) cho mỗi context
            - Nếu input là 1 context: trả về list[(q1, a1), (q2, a2), ...]
            - Nếu input là list context: trả về [list1, list2, ...]
        """
        logging.info(f'running model for `question_answer_pair_generation`')
        assert self.is_qag, "`generate_qa_end2end` is available for end2end_qag_model"
        
        # Xác định prefix task nếu model yêu cầu
        prefix_type = 'qag' if self.add_prefix else None
        
        # Chuẩn hóa input: chuyển string thành list để xử lý thống nhất
        single_input = type(list_context) is str
        list_context = [list_context] if single_input else list_context
        
        # Gọi model để generate output string
        output = self.generate_prediction(
            list_context, prefix_type=prefix_type, cache_path=cache_path, num_beams=num_beams, batch_size=batch_size
        )

        def format_qa(list_raw_string):
            """Parse chuỗi output thành danh sách cặp (question, answer).
            
            Ví dụ input: "question: Ai?, answer: Bạn"
            Output: ("Ai?", "Bạn")
            """
            tmp = []
            for raw_string in list_raw_string:
                # Kiểm tra format có hợp lệ không (phải có answer_prefix và question_prefix)
                if len(raw_string.split(answer_prefix)) != 2 or question_prefix not in raw_string:
                    logging.info(f"invalid prediction: {raw_string}")
                else:
                    # Tách question và answer
                    q, a = raw_string.split(answer_prefix)
                    # Làm sạch khoảng trắng thừa ở answer
                    a = re.sub(r'\A\s+', '', a)
                    a = re.sub(r'\s+\Z', '', a)
                    # Bỏ question_prefix và làm sạch question
                    q = q.replace(question_prefix, "")
                    q = re.sub(r'\A\s+', '', q)
                    q = re.sub(r'\s+\Z', '', q)
                    tmp.append((q, a))
            return tmp

        # Parse output string thành danh sách cặp (question, answer)
        # Mỗi context có thể sinh ra nhiều cặp QA, phân tách bằng splitting_symbol
        output = [format_qa(o.split(splitting_symbol)) for o in output]
        
        # Nếu input ban đầu là 1 string -> trả về 1 list, không phải list của list
        return output[0] if single_input else output

    def generate_qa(self,
                    list_context: str or List,
                    batch_size: int = None,
                    num_beams: int = 4,
                    cache_path: str = None,
                    num_questions: int = None,
                    sentence_level: bool = False):
        """Sinh cặp QA từ context.

        Luồng xử lý:
        - Nếu model là QAG end-to-end: gọi `generate_qa_end2end` trực tiếp
        - Nếu là pipeline: chạy AE trước để tìm answer, sau đó chạy QG cho từng answer
        
        Args:
            list_context: Văn bản đầu vào (1 context hoặc list)
            batch_size: Batch size cho inference
            num_beams: Số beam search
            cache_path: Đường dẫn cache feature đã encode
            num_questions: Giới hạn số câu hỏi (chủ yếu cho spaCy AE)
            sentence_level: Bật prediction theo câu để giảm độ phức tạp
            
        Returns:
            Danh sách cặp (question, answer) cho mỗi context
            - Nếu không tìm thấy answer nào: trả về None cho context đó
        """
        # Nếu model hỗ trợ QAG end-to-end -> gọi trực tiếp
        if self.is_qag:
            return self.generate_qa_end2end(list_context, batch_size, num_beams, cache_path)
        
        # Chuẩn hóa input
        single_input = type(list_context) is str
        list_context = [list_context] if single_input else list_context
        original_input_length = len(list_context)

        # Bước 1: Chạy Answer Extraction để tìm các câu trả lời tiềm năng
        logging.info('running model for `ae`')
        list_answer = self.generate_a(
            list_context,
            batch_size=batch_size,
            num_beams=num_beams,
            cache_path=cache_path,
            sentence_level=sentence_level,
            num_questions=num_questions
        )
        
        # Bước 2: Lọc ra các context có tìm thấy answer (bỏ qua những cái None)
        valid_context_id = [n for n, a in enumerate(list_answer) if a is not None]
        list_context = [list_context[n] for n in valid_context_id]
        list_answer = [list_answer[n] for n in valid_context_id]
        
        # Bước 3: Chuẩn bị input cho QG
        # Mỗi context có nhiều answer -> tạo nhiều input (context, answer) cho QG
        qg_input, qg_hl, list_length = [], [], [0]
        for c, a in zip(list_context, list_answer):
            qg_hl += a  # Danh sách tất cả answer (sẽ dùng làm highlight)
            qg_input += [c] * len(a)  # Nhân bản context theo số answer
            list_length.append(list_length[-1] + len(a))  # Đánh dấu ranh giới  # Đánh dấu ranh giới
        
        # Bước 4: Chạy Question Generation cho từng cặp (context, answer)
        logging.info('running model for `qg`')
        list_question = self.generate_q(
            qg_input,
            list_answer=qg_hl,  # Highlight answer trong context
            batch_size=batch_size,
            cache_path=cache_path,
            num_beams=num_beams,
            sentence_level=sentence_level
        )

        assert len(qg_hl) == len(list_question), f"{len(qg_input)} != {len(list_question)}"

        # Bước 5: Gồm kết quả về đúng cấu trúc theo từng context ban đầu
        # list_length chứa ranh giới để biết question/answer nào thuộc context nào
        list_question = [list_question[list_length[n - 1]:list_length[n]] for n in range(1, len(list_length))]
        list_answer = [qg_hl[list_length[n - 1]:list_length[n]] for n in range(1, len(list_length))]
        
        # Bước 6: Tạo output list với đầy đủ các context (kể cả những cái không tìm thấy answer)
        output_list = [None] * original_input_length
        # Điền kết quả vào đúng vị trí ban đầu
        for n, _id in enumerate(valid_context_id):
            output_list[_id] = [(q, a) for q, a in zip(list_question[n], list_answer[n])]
        
        # Trả về kết quả: unwrap nếu input ban đầu là single string
        return output_list[0] if single_input else output_list

    def generate_a(self,
                   context: str or List,
                   batch_size: int = None,
                   num_beams: int = 4,
                   cache_path: str = None,
                   sentence_level: bool = False,
                   num_questions: int = None):
        """Trích xuất đáp án (answer) từ context.

        Có 3 cách trích xuất answer tùy thuộc vào backend:
        1. spaCy: Dùng thuật toán keyword extraction (positionrank, textrank, yake, v.v.)
        2. Multitask model: Dùng model chính đã fine-tune cho AE task
        3. Pipeline model: Dùng model AE riêng biệt
        
        Args:
            context: Văn bản đầu vào (1 context hoặc list)
            batch_size: Batch size cho inference
            num_beams: Số beam search (cho model-based AE)
            cache_path: Đường dẫn cache feature đã encode
            sentence_level: Bật prediction theo từng câu (giảm độ phức tạp)
            num_questions: Số đáp án tối đa cần trích xuất (chủ yếu cho spaCy backend)
            
        Returns:
            Danh sách đáp án cho mỗi context
            - Mỗi context -> list các answer string
            - Nếu không tìm thấy answer nào -> None
        """
        logging.info(f'running model for `answer_extraction`')
        
        # Nếu dùng spaCy backend -> gọi trực tiếp spaCy keyword extraction
        if self.answer_model_type == 'spacy':
            num_questions = 10 if num_questions is None else num_questions
            if type(context) is str:
                return self.spacy_module.keyword(context, num_questions)
            else:
                return [self.spacy_module.keyword(c, num_questions) for c in context]
        
        # Nếu dùng model-based AE (multitask hoặc pipeline)
        # Chuẩn hóa input
        single_input = type(context) is str
        context = [context] if single_input else context
        
        # Bước 1: Tách context thành danh sách câu (sentence segmentation)
        # Ví dụ: "Hà Nội là thủ đô. Việt Nam ở châu Á." -> ["Hà Nội là thủ đô.", "Việt Nam ở châu Á."]
        list_sentences = [self.spacy_module.sentence(c) for c in context]
        
        # Nếu sentence_level=False: input là (context, sentence) để model biết cần extract answer từ câu nào
        # Nếu sentence_level=True: input chỉ là câu (giảm độ phức tạp)
        list_inputs = [[c] * len(s) for c, s in zip(context, list_sentences)]  # Nhân bản context
        list_length = [0] + np.cumsum([len(s) for s in list_sentences]).tolist()  # Ranh giới
        
        if sentence_level:
            list_inputs = list_sentences  # Chỉ dùng câu thay vì (context + câu)  # Chỉ dùng câu thay vì (context + câu)
        
        # Bước 2: Flatten để đưa vào batch generate
        flat_sentences = list(chain(*list_sentences))  # Tất cả các câu
        flat_inputs = list(chain(*list_inputs))  # Tất cả input tương ứng
        
        # Bước 3: Gọi model AE để generate answer
        if self.answer_model_type == 'multitask':
            # Dùng model chính (multitask model có cả AE và QG)
            answer = self.generate_prediction(
                flat_inputs,
                highlights=flat_sentences,  # Highlight câu cần extract
                prefix_type='ae' if self.add_prefix else None,
                cache_path=cache_path,
                num_beams=num_beams,
                batch_size=batch_size
            )
        elif self.answer_model_type == 'pipeline':
            # Dùng model AE riêng biệt
            answer = self.generate_prediction(
                flat_inputs,
                highlights=flat_sentences,
                prefix_type='ae' if self.add_prefix_ae else None,
                cache_path=cache_path,
                num_beams=num_beams,
                batch_size=batch_size,
                switch_to_model_ae=True  # Chuyển sang dùng model_ae
            )
        else:
            raise ValueError(f"unknown answer model type: {self.answer_model_type}")
        
        # Bước 4: Khôi phục lại cấu trúc nested theo context ban đầu
        # Làm sạch khoảng trắng thừa
        answer = [clean(a) for a in answer]
        
        # Chia answer theo ranh giới list_length
        list_answer = [answer[list_length[n - 1]:list_length[n]] for n in range(1, len(list_length))]
        
        # Lọc chỉ giữ những answer thực sự nằm trong context/sentence
        # (Bỏ qua answer None hoặc không tìm thấy trong context)
        list_answer = [[a for a, c in zip(a_sent, c_sent) if a is not None and a in c]
                       for a_sent, c_sent in zip(list_answer, list_inputs)]
        
        # Đánh dấu None cho context không tìm thấy answer nào
        list_answer = [None if len(a) == 0 else a for a in list_answer]
        
        # Nếu drop_answer_error_text=False -> raise error khi không tìm thấy answer
        if not self.drop_answer_error_text:
            if any(a is None for a in list_answer):
                raise AnswerNotFoundError([context[n] for n, a in enumerate(list_answer) if a is None][0])
        
        # Trả về kết quả
        return list_answer[0] if single_input else list_answer

    def generate_q(self,
                   list_context: str or List,
                   list_answer: List = None,
                   batch_size: int = None,
                   num_beams: int = 4,
                   cache_path: str = None,
                   sentence_level: bool = False):
        """Sinh câu hỏi từ context và answer (đáp án được highlight bằng `<hl>`).

        Model QG nhận input dạng: "Văn bản trước <hl> câu trả lời <hl> văn bản sau"
        Và sinh ra câu hỏi tương ứng với câu trả lời đó.
        
        Args:
            list_context: Context đầu vào (1 context hoặc list)
            list_answer: Danh sách answer cần highlight trong context
            batch_size: Batch size cho inference
            num_beams: Số beam search
            cache_path: Đường dẫn cache feature đã encode
            sentence_level: Bật prediction theo câu để giảm độ phức tạp
            
        Returns:
            Câu hỏi sinh ra (string hoặc list string)
        """
        assert self.is_qg, "model is not fine-tuned for QG"
        
        # Kiểm tra type consistency
        if list_answer is not None:
            assert type(list_context) is type(list_answer), f"{type(list_context)} != {type(list_answer)}"
        
        # Chuẩn hóa input
        single_input = False
        if type(list_context) is str:
            list_context = [list_context]
            list_answer = [list_answer] if list_answer is not None else None
            single_input = True
        
        # Gọi generate_prediction với highlight = answer
        output = self.generate_prediction(
            list_context,
            highlights=list_answer,  # Token <hl> sẽ được chèn quanh answer
            prefix_type='qg' if self.add_prefix else None,
            cache_path=cache_path,
            num_beams=num_beams,
            batch_size=batch_size,
            sentence_level=sentence_level
        )
        
        # Trả về kết quả
        if single_input:
            return output[0]
        return output

    def answer_q(self,
                 list_context: str or List,
                 list_question: str or List,
                 batch_size: int = None,
                 num_beams: int = 4,
                 cache_path: str = None):
        """Trả lời câu hỏi dựa trên context cho sẵn.
        
        Model QA nhận input dạng: "question: <câu hỏi>, context: <văn bản>"
        Và sinh ra câu trả lời.
        
        Args:
            list_context: Văn bản chứa câu trả lời (1 hoặc list)
            list_question: Câu hỏi cần trả lời (1 hoặc list)
            batch_size: Batch size cho inference
            num_beams: Số beam search
            cache_path: Đường dẫn cache feature đã encode
            
        Returns:
            Câu trả lời (string hoặc list string)
        """
        logging.info(f'running model for `question_answering`')
        assert self.is_qa, "model is not fine-tuned for QA"
        assert type(list_context) is type(list_question), "invalid input"
        
        # Chuẩn hóa input
        single_input = type(list_context) is str
        list_context = [list_context] if single_input else list_context
        list_question = [list_question] if single_input else list_question
        assert len(list_context) == len(list_question), f"invalid input: {len(list_context)} != {len(list_question)}"
        
        # Tạo input string dạng: "question: ..., context: ..."
        output = self.generate_prediction(
            [f"question: {q}, context: {c}" for q, c in zip(list_question, list_context)],
            batch_size=batch_size,
            prefix_type='qa' if self.add_prefix else None,
            cache_path=cache_path,
            num_beams=num_beams
        )
        
        return output[0] if single_input else output

    def generate_prediction(self,
                            inputs: List,
                            highlights: List or None = None,
                            prefix_type: str = None,
                            num_beams: int = 4,
                            batch_size: int = None,
                            cache_path: str = None,
                            sentence_level: bool = False,
                            switch_to_model_ae: bool = False):
        """Hàm generate tổng quát cho QG/AE/QA - core inference method.

        Đây là hàm chính thực hiện inference cho tất cả các tác vụ.
        Luồng xử lý:
        1. Tokenize input text (có thể kèm highlight)
        2. Tạo DataLoader cho batch processing
        3. Chạy model.generate() với beam search
        4. Decode output tokens thành text
        
        Args:
            inputs: Danh sách input text
            highlights: Danh sách span cần highlight bằng `<hl>` (thường là answer)
            prefix_type: Prefix tác vụ ('qg', 'ae', 'qag', 'qa')
            num_beams: Số beam search (mặc định 4)
            batch_size: Batch size cho inference
            cache_path: Đường dẫn cache feature đã encode (tiết kiệm thời gian)
            sentence_level: Chỉ xử lý ở mức câu (giảm độ phức tạp)
            switch_to_model_ae: Dùng model_ae thay vì model chính
            
        Returns:
            Danh sách chuỗi đã generate
        """
        # Chuyển model sang eval mode (tắt dropout, batch norm không update, v.v.)
        self.eval()
        
        # Chọn model và tokenizer: model chính hoặc model_ae
        if switch_to_model_ae:
            assert self.model_ae is not None and self.tokenizer_ae is not None
            model = self.model_ae
            tokenizer = self.tokenizer_ae
            max_length_output = self.max_length_output_ae
        else:
            model = self.model
            tokenizer = self.tokenizer
            max_length_output = self.max_length_output

        # Nếu sentence_level=True: chỉ xử lý câu chứa answer thay vì toàn bộ context
        # Điều này giảm độ phức tạp và tăng tốc độ
        if sentence_level:
            assert highlights is not None, '`sentence_level` cần tham số `highlights` để xác định câu chứa answer.'
            assert len(highlights) == len(inputs), str([len(highlights), len(inputs)])
            list_sentence = []
            for context, answer in zip(inputs, highlights):
                # Tìm câu chứa answer
                s = [sentence for sentence in self.spacy_module.sentence(context) if answer in sentence]
                list_sentence.append(s[0] if len(s) != 0 else context)  # Fallback về context nếu không tìm thấy
            inputs = list_sentence

        assert type(inputs) is list, inputs
        
        # Bước 1: Tokenize tất cả input text (với highlight nếu có)
        encode_list = self.text_to_encode(
            inputs,
            highlights=highlights,
            prefix_type=prefix_type,
            cache_path=cache_path,
            switch_to_model_ae=switch_to_model_ae
        )
        
        # Bước 2: Tạo DataLoader cho batch processing
        loader = self.get_data_loader(encode_list, batch_size=batch_size)
        
        # Bước 3: Lặp qua từng batch và generate
        outputs = []
        for encode in loader:
            with torch.no_grad():  # Không tính gradient (tiết kiệm bộ nhớ)
                # Bỏ labels nếu có (không cần cho inference)
                if 'labels' in encode:
                    encode.pop('labels')
                
                # Chuyển tensor lên device (GPU/CPU)
                encode = {k: v.to(self.device) for k, v in encode.items()}
                
                # Thêm tham số generate
                encode['max_length'] = max_length_output
                encode['num_beams'] = num_beams
                
                # Gọi model.generate() (unwrap nếu dùng DataParallel)
                tensor = model.module.generate(**encode) if self.parallel else model.generate(**encode)
                
                # Decode token IDs thành text
                outputs += tokenizer.batch_decode(tensor, skip_special_tokens=True)
        
        return outputs

    def encode_to_loss(self, encode: Dict):
        """Tính loss từ feature đã encode (dùng cho fine-tuning).

        Hàm này được gọi trong training loop để tính loss.
        Nếu có label_smoothing, sử dụng custom loss function,
        không thì dùng loss tuộng trong model.
        
        Args:
            encode: Dict feature đã tokenize (phải có `labels`)
            
        Returns:
            Giá trị loss scalar (single number)
        """
        assert 'labels' in encode
        
        # Forward pass qua model
        output = self.model(**{k: v.to(self.device) for k, v in encode.items()})
        
        # Tính loss
        if self.label_smoothing is None or self.label_smoothing == 0.0:
            # Dùng loss mặc định của model (cross-entropy)
            # Nếu DataParallel -> lấy trung bình loss từ các GPU
            return output['loss'].mean() if self.parallel else output['loss']
        else:
            # Dùng custom label smoothing loss
            return label_smoothed_loss(output['logits'], encode['labels'].to(self.device), self.label_smoothing)

    def text_to_encode(self,
                       inputs,
                       outputs: List = None,
                       highlights: List = None,
                       prefix_type: str = None,
                       cache_path: str = None,
                       switch_to_model_ae: bool = False):
        """Chuyển text đầu vào/đầu ra thành feature tokenized.

        Luồng xử lý:
        1. Kiểm tra cache: nếu đã encode trước đó -> load từ cache
        2. Tokenize tất cả sample (có thể dùng multiprocessing)
        3. Lọc bỏ các sample lỗi (overlength, highlight không tìm thấy)
        4. Lưu cache cho lần sau
        
        Args:
            inputs: Danh sách input text
            outputs: Danh sách output tương ứng (nếu có, cho training)
            highlights: Danh sách span cần đánh dấu `<hl>`
            prefix_type: Prefix tác vụ
            cache_path: Đường dẫn cache feature trung gian
            switch_to_model_ae: Dùng tokenizer_ae và config của model_ae
            
        Returns:
            Danh sách feature đã encode (dict có input_ids, attention_mask, labels)
        """
        # Bước 1: Kiểm tra cache
        if cache_path is not None and os.path.exists(cache_path):
            logging.info(f'loading preprocessed feature from {cache_path}')
            return pickle_load(cache_path)
        
        # Bước 2: Chuẩn bị data cho encoding
        # Đảm bảo outputs và highlights có cùng độ dài với inputs
        outputs = [None] * len(inputs) if outputs is None else outputs
        highlights = [None] * len(inputs) if highlights is None else highlights
        assert len(outputs) == len(inputs) == len(highlights), str([len(outputs), len(inputs), len(highlights)])
        
        # Zip thành danh sách tuple (input, output, highlight)
        data = list(zip(inputs, outputs, highlights))
        
        # Bước 3: Tạo EncodePlus object với cấu hình phù hợp
        config = {'tokenizer': self.tokenizer, 'max_length': self.max_length, 'prefix_type': prefix_type,
                  'max_length_output': self.max_length_output, 'drop_overflow_error_text': self.drop_overflow_error_text,
                  'skip_overflow_error': self.skip_overflow_error, 'drop_highlight_error_text': self.drop_highlight_error_text,
                  'padding': False if len(data) == 1 else True}  # padding=True cho batch, False cho single
        
        # Nếu dùng model_ae -> thay đổi config
        if switch_to_model_ae:
            assert self.model_ae is not None and self.tokenizer_ae is not None
            config['tokenizer'] = self.tokenizer_ae
            config['max_length'] = self.max_length_ae
            config['max_length_output'] = self.max_length_output_ae

        logging.info(f'encode all the data       : {len(data)}')
        
        # Tạo thư mục cache nếu cần
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Bước 4: Chọn cách xử lý: song song (multiprocessing) hoặc đơn luồng
        if PARALLEL_PROCESSING:
            # Dùng multiprocessing Pool để tăng tốc
            pool = Pool()
            out = pool.map(EncodePlus(**config), data)
            pool.close()
            out = list(filter(None, out))  # Loại bỏ các mẫu bị drop (overlength/highlight lỗi)
        else:
            # Xử lý đơn luồng (dễ debug hơn)
            f = EncodePlus(**config)
            out = []
            files = []  # Danh sách file cache tạm
            
            for i in tqdm(data):
                e = f(i)
                if e is not None:  # Chỉ giữ các mẫu hợp lệ
                    out.append(e)
                
                # Lưu cache tạm thời sau mỗi 40k sample (tránh out of memory)
                if len(out) > 40000 and cache_path is not None:
                    pickle_save(out, f'{cache_path}.tmp{len(files)}')
                    files.append(f'{cache_path}.tmp{len(files)}')
                    out = []
            
            # Lưu batch cuối cùng
            if len(out) > 0 and cache_path is not None:
                pickle_save(out, f'{cache_path}.tmp{len(files)}')
                files.append(f'{cache_path}.tmp{len(files)}')
            
            # Gộp tất cả file tạm lại
            if len(files) > 0:
                out = list(chain(*[pickle_load(i) for i in files]))
        
        logging.info(f'after remove the overflow : {len(out)}')
        
        # Bước 5: Lưu cache để tái sử dụng ở lần chạy sau
        if cache_path is not None:
            pickle_save(out, cache_path)
            logging.info(f'preprocessed feature is saved at {cache_path}')
        
        return out

    def save(self, save_dir):
        """Lưu model và tokenizer ra thư mục.

        Lưu cả:
        - Model weights (pytorch_model.bin)
        - Config (config.json)
        - Tokenizer (tokenizer_config.json, vocab, v.v.)
        
        Args:
            save_dir: Thư mục đích để lưu
        """

        def model_state(model):
            """Lấy model thực sự (unwrap DataParallel nếu có)."""
            if self.parallel:
                return model.module  # DataParallel wrap model trong .module
            return model

        logging.info('saving model')
        # Cập nhật config với add_prefix flag
        model_state(self.model).config.update({'add_prefix': self.add_prefix})
        # Lưu model
        model_state(self.model).save_pretrained(save_dir)
        
        logging.info('saving tokenizer')
        # Lưu tokenizer
        self.tokenizer.save_pretrained(save_dir)

    @staticmethod
    def get_data_loader(encode_list, batch_size: int = None, shuffle: bool = False, drop_last: bool = False):
        """Tạo DataLoader từ danh sách feature đã encode.

        DataLoader tự động:
        - Chia data thành batch
        - Shuffle data (nếu shuffle=True)
        - Tải data song song với num_workers
        
        Args:
            encode_list: Danh sách feature đã encode (list các dict)
            batch_size: Batch size (Nếu None -> lấy toàn bộ data làm 1 batch)
            shuffle: Trộn dữ liệu trước mỗi epoch (dùng cho training)
            drop_last: Bỏ batch cuối nếu không đủ số lượng (dùng cho training)
            
        Returns:
            torch.utils.data.DataLoader object
        """
        # Nếu không chỉ định batch_size -> dùng toàn bộ data làm 1 batch
        batch_size = len(encode_list) if batch_size is None else batch_size
        
        # Tham số cho DataLoader
        params = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=NUM_WORKERS)
        
        # Tạo và trả về DataLoader
        return torch.utils.data.DataLoader(Dataset(encode_list), **params)

    def train(self):
        """Chuyển model sang training mode.
        
        Training mode:
        - Bật dropout
        - Batch normalization sẽ update running stats
        - Gradient sẽ được tính
        """
        self.model.train()

    def eval(self):
        """Chuyển model sang evaluation mode.
        
        Evaluation mode:
        - Tắt dropout
        - Batch normalization dùng running stats (không update)
        - Thường kết hợp với torch.no_grad() để không tính gradient
        """
        self.model.eval()
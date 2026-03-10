import os
import logging
import pickle
import re
import urllib
from itertools import chain
from typing import List, Dict
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional
import transformers
from .exceptions import ExceedMaxLengthError, HighlightNotFoundError, AnswerNotFoundError
from .spacy_module import SpacyPipeline, VALID_METHODS

__all__ = ('TransformersQG', 'ADDITIONAL_SP_TOKENS', 'TASK_PREFIX', 'clean', 'internet_connection')

# Tat canh bao parallelism cua tokenizer de log gon hon.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
TASK_PREFIX = {
    "ae": "extract answers",
    "qg": "generate question",
    "qag": "generate question and answer",
    "qa": "answer question"
}
CE_IGNORE_INDEX = -100
ADDITIONAL_SP_TOKENS = {'hl': '<hl>'}
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '0'))
PARALLEL_PROCESSING = bool(int(os.getenv('PARALLEL_PROCESSING', '0')))
DEFAULT_MODELS = {
    'vi': 'VietAI/vit5-base'
}


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Doc lai object da duoc pickle truoc do.
        return pickle.load(fp)


def clean(string):
    string = re.sub(r'\A\s*', '', string)
    string = re.sub(r'\s*\Z', '', string)
    if len(string) > 0:
        return string
    return None


def internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


def load_language_model(model_name,
                        cache_dir: str = None,
                        use_auth_token: bool = False,
                        torch_dtype=None,
                        device_map: str = None,
                        low_cpu_mem_usage: bool = False):
    """Nap tokenizer/model tu Hugging Face va bo sung special token can thiet.

    - Tu dong chon `local_files_only=True` khi khong co internet.
    - Chon dung class model theo `config.model_type`.
    - Them token `<hl>` de danh dau span highlight trong input.
    """
    # Tokenizer + config
    local_files_only = not internet_connection()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only, use_auth_token=use_auth_token)
    config = transformers.AutoConfig.from_pretrained(
        model_name, local_files_only=local_files_only, cache_dir=cache_dir, use_auth_token=use_auth_token)
    # Chon class model theo loai architecture.
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

    param = {'config': config, "local_files_only": local_files_only, "use_auth_token": use_auth_token,
             "low_cpu_mem_usage": low_cpu_mem_usage, "cache_dir": cache_dir}
    if torch_dtype is not None:
        param['torch_dtype'] = torch_dtype
    if device_map is not None:
        param['device_map'] = device_map
    model = model_class(model_name, **param)
    # Dong bo tokenizer/model voi token dac biet (<hl>) dung trong pipeline QG.
    tokenizer.add_special_tokens({'additional_special_tokens': list(ADDITIONAL_SP_TOKENS.values())})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model, config


def label_smoothed_loss(logits, labels, epsilon):
    """ https://github.com/huggingface/transformers/blob/55bb4c06f7be141c6d895dbe1f11018dc8580b2d/src/transformers/trainer_pt_utils.py#L430 """
    log_probs = - functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(CE_IGNORE_INDEX)
    # Neu label = -100 thi gather se loi, nen tam clamp ve 0.
    # Cac vi tri pad van duoc loai bo boi `padding_mask`.
    labels.clamp_min_(0)

    nll_loss = log_probs.gather(dim=-1, index=labels)
    nll_loss.masked_fill_(padding_mask, 0.0)

    # Cong thuc nay van on voi fp16 vi tong duoc tinh o fp32.
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Chuan hoa theo so token hop le (khong tinh padding).
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


class Dataset(torch.utils.data.Dataset):
    """Wrapper nho cho DataLoader: chuyen dict feature -> tensor."""
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class EncodePlus:
    """Wrapper cho buoc tokenize, co the dung trong multiprocessing."""

    def __init__(self,
                 tokenizer,
                 max_length: int = 512,
                 max_length_output: int = 34,
                 drop_overflow_error_text: bool = False,
                 skip_overflow_error: bool = False,
                 drop_highlight_error_text: bool = False,
                 prefix_type: str = None,
                 padding: bool = True):
        """Khoi tao bo ma hoa input/output.

        @param tokenizer: transforms.Tokenizer
        @param max_length: Do dai toi da cua input.
        @param max_length_output: Do dai toi da cua output.
        @param drop_overflow_error_text: True -> bo mau vuot max length (tra ve None).
        @param skip_overflow_error: True -> bo qua check overflow.
        @param drop_highlight_error_text: True -> bo mau neu khong tim thay highlight span.
        @param prefix_type: Prefix tac vu (`qg`, `ae`, `qag`, `qa`).
        @param padding: Co pad ve max_length hay khong.
        """
        self.prefix = TASK_PREFIX[prefix_type] if prefix_type is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_output = max_length_output
        # Khi train thuong bo mau overlength; khi evaluate co the muon giu lai.
        self.drop_overflow_error_text = drop_overflow_error_text
        self.skip_overflow_error = skip_overflow_error
        self.drop_highlight_error_text = drop_highlight_error_text
        # Bat truncation cho pipeline batch de tranh vo batch.
        self.param_in = {'truncation': True, 'max_length': self.max_length}
        self.param_out = {'truncation': True, 'max_length': self.max_length_output}
        if padding:
            self.param_in['padding'] = 'max_length'
            self.param_out['padding'] = 'max_length'

    def __call__(self, inputs):
        return self.encode_plus(*inputs)

    def encode_plus(self, input_sequence: str, output_sequence: str = None, input_highlight: str = None):
        """Tokenize 1 mau du lieu.

        @param input_sequence: Cau/van ban dau vao.
        @param output_sequence: Chuoi muc tieu (neu co, dung cho train).
        @param input_highlight: Chuoi con can danh dau bang `<hl>`.
        @return: Dict feature da tokenized (co the kem `labels`).
        """
        # Chen token <hl> quanh span cau tra loi de model biet vi tri can hoi.
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

        # Xu ly overlength:
        # - drop_overflow_error_text=True: loai bo mau.
        # - skip_overflow_error=True: bo qua check.
        # - mac dinh: nem exception de phat hien du lieu loi.
        if self.drop_overflow_error_text or not self.skip_overflow_error:
            if len(self.tokenizer.encode(input_sequence)) > self.max_length:
                if not self.drop_overflow_error_text:  # Bao loi neu khong cho phep drop.
                    raise ExceedMaxLengthError(self.max_length)
                return None  # Loai bo mau overlength.
            if output_sequence is not None:
                if len(self.tokenizer.encode(output_sequence)) > self.max_length_output:
                    if not self.drop_overflow_error_text:  # Bao loi neu output qua dai.
                        raise ExceedMaxLengthError(self.max_length)
                    return None  # Loai bo mau overlength.
        if type(self.tokenizer) is transformers.models.mbart.tokenization_mbart_fast.MBartTokenizerFast:
            encode = self.tokenizer(input_sequence, **self.param_in)
        else:
            encode = self.tokenizer(text_target=input_sequence, **self.param_in)
        if output_sequence is not None:
            encode['labels'] = self.tokenizer.encode(output_sequence, **self.param_out)
        return encode


class TransformersQG:
    """Model wrapper cho cac tac vu QG/AE/QA/QAG tren Transformers."""

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
        """Khoi tao model va cac thanh phan phu tro cho sinh cau hoi.

        @param model: Ten model tren hub hoac duong dan local.
        @param max_length: Do dai toi da input.
        @param max_length_output: Do dai toi da output.
        @param cache_dir: Thu muc cache model/tokenizer.
        @param add_prefix: Co them prefix tac vu vao input hay khong.
        @param language: Ngon ngu dung cho pipeline spaCy.
        @param label_smoothing: He so label smoothing khi train.
        @param drop_overflow_error_text: True -> bo mau overlength.
        @param skip_overflow_error: True -> bo qua check overlength.
        @param drop_highlight_error_text: True -> bo mau neu khong tim thay highlight.
        @param use_auth_token: [optional] Huggingface transformers argument of `use_auth_token`
        """

        # Neu khong truyen model, lay model mac dinh theo ngon ngu.
        if model is None:
            assert language in DEFAULT_MODELS.keys(),\
                f"Model with language '{language}' is not available. Please choose language from " \
                f"'{DEFAULT_MODELS.keys()}' or specify 'model'."
            model = DEFAULT_MODELS[language]

        # Suy ra kha nang tac vu tu ten model (qg/ae/qa/qag).
        self.is_qg = 'qg' in model.split('-') if is_qg is None else is_qg
        self.is_ae = 'ae' in model.split('-') if is_ae is None else is_ae
        self.is_qa = 'qa' in model.split('-') if is_qa is None else is_qa
        self.is_qag = 'qag' in model.split('-') if is_qag is None else is_qag
        # Luu cau hinh co ban.
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
        # Nap model chinh (QG/QA/QAG).
        self.tokenizer, self.model, config = load_language_model(
            self.model_name, cache_dir=cache_dir, use_auth_token=use_auth_token, device_map=device_map,
            torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        if 'add_prefix' not in config.to_dict().keys():
            # Model chua co flag add_prefix trong config (thuong la base model).
            # assert add_prefix, '`add_prefix` is required for non-fine-tuned models'
            self.add_prefix = add_prefix
        else:
            self.add_prefix = config.add_prefix

        # Cai dat mac dinh cho module trich xuat dap an (AE).
        if self.model_name_ae is None:
            self.model_name_ae = self.model_name if self.is_ae else "positionrank"
        # Chon backend AE: spaCy / multitask (chung model) / pipeline (model rieng).
        self.answer_model_type = None
        if self.model_name_ae in VALID_METHODS:
            logging.info(f'use spaCy answer extraction model: {self.model_name_ae}')
            self.tokenizer_ae = self.model_ae = self.add_prefix_ae = None
            self.spacy_module = SpacyPipeline(language, self.model_name_ae)
            self.answer_model_type = 'spacy'
        else:
            logging.info(f'use LMQG fine-tuned answer extraction model: {self.model_name_ae}')
            if self.model_name == self.model_name_ae:
                logging.info("the same model as QG is used as AE")
                assert self.is_ae, f"the model ({self.model_name_ae}) is not fine-tuned for AE"
                self.tokenizer_ae = self.model_ae = self.add_prefix_ae = None
                self.answer_model_type = 'multitask'
            else:
                logging.info(f"loading 2nd model for AE: {self.model_name_ae}")
                self.tokenizer_ae, self.model_ae, config_ae = load_language_model(model_ae, cache_dir=cache_dir, use_auth_token=use_auth_token)
                self.add_prefix_ae = config_ae.add_prefix
                self.answer_model_type = 'pipeline'
            self.spacy_module = SpacyPipeline(language)

        # Thiet lap thiet bi tinh toan va DataParallel neu nhieu GPU.
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
            if self.model_ae is not None:
                self.model_ae = torch.nn.DataParallel(self.model_ae)
        self.model.to(self.device)
        if self.model_ae is not None:
            self.model_ae.to(self.device)
        logging.info(f'Model `{self.model_name}`')
        logging.info(f'\t * Num of GPU in use: {torch.cuda.device_count()}')
        logging.info(f'\t * Prefix: {self.add_prefix}')
        logging.info(f'\t * Language: {language} (ignore at the training phase)')

    def push_to_hub(self, repo_id):
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
        """Sinh truc tiep danh sach (question, answer) tu context bang model QAG end-to-end.

        @param list_context: Mot context hoac danh sach context.
        @param batch_size: Batch size khi generate.
        @param num_beams: So beam search.
        @param cache_path: Duong dan cache feature da encode.
        @return: Danh sach cap (question, answer) cho moi context.
        """
        logging.info(f'running model for `question_answer_pair_generation`')
        assert self.is_qag, "`generate_qa_end2end` is available for end2end_qag_model"
        prefix_type = 'qag' if self.add_prefix else None
        single_input = type(list_context) is str
        list_context = [list_context] if single_input else list_context
        output = self.generate_prediction(
            list_context, prefix_type=prefix_type, cache_path=cache_path, num_beams=num_beams, batch_size=batch_size
        )

        def format_qa(list_raw_string):
            tmp = []
            for raw_string in list_raw_string:
                if len(raw_string.split(answer_prefix)) != 2 or question_prefix not in raw_string:
                    logging.info(f"invalid prediction: {raw_string}")
                else:
                    q, a = raw_string.split(answer_prefix)
                    a = re.sub(r'\A\s+', '', a)
                    a = re.sub(r'\s+\Z', '', a)
                    q = q.replace(question_prefix, "")
                    q = re.sub(r'\A\s+', '', q)
                    q = re.sub(r'\s+\Z', '', q)
                    tmp.append((q, a))
            return tmp

        output = [format_qa(o.split(splitting_symbol)) for o in output]
        return output[0] if single_input else output

    def generate_qa(self,
                    list_context: str or List,
                    batch_size: int = None,
                    num_beams: int = 4,
                    cache_path: str = None,
                    num_questions: int = None,
                    sentence_level: bool = False):
        """Sinh cap QA tu context.

        - Neu model la QAG end-to-end: goi `generate_qa_end2end`.
        - Neu la pipeline: AE truoc, sau do QG cho tung answer.

        @param list_context: Van ban dau vao.
        @param batch_size: Batch size.
        @param num_beams: So beam search.
        @param cache_path: Duong dan cache feature da encode.
        @param num_questions: Gioi han so cau hoi (chu yeu cho spaCy AE).
        @param sentence_level: Bat prediction theo cau de giam do phuc tap.
        @return: Danh sach cap (question, answer).
        """
        if self.is_qag:
            return self.generate_qa_end2end(list_context, batch_size, num_beams, cache_path)
        single_input = type(list_context) is str
        list_context = [list_context] if single_input else list_context
        original_input_length = len(list_context)

        logging.info('running model for `ae`')
        list_answer = self.generate_a(
            list_context,
            batch_size=batch_size,
            num_beams=num_beams,
            cache_path=cache_path,
            sentence_level=sentence_level,
            num_questions=num_questions
        )
        valid_context_id = [n for n, a in enumerate(list_answer) if a is not None]
        list_context = [list_context[n] for n in valid_context_id]
        list_answer = [list_answer[n] for n in valid_context_id]
        qg_input, qg_hl, list_length = [], [], [0]
        for c, a in zip(list_context, list_answer):
            qg_hl += a
            qg_input += [c] * len(a)
            list_length.append(list_length[-1] + len(a))
        logging.info('running model for `qg`')
        list_question = self.generate_q(
            qg_input,
            list_answer=qg_hl,
            batch_size=batch_size,
            cache_path=cache_path,
            num_beams=num_beams,
            sentence_level=sentence_level
        )

        assert len(qg_hl) == len(list_question), f"{len(qg_input)} != {len(list_question)}"

        # Gom ket qua ve dung cau truc theo tung context ban dau.
        list_question = [list_question[list_length[n - 1]:list_length[n]] for n in range(1, len(list_length))]
        list_answer = [qg_hl[list_length[n - 1]:list_length[n]] for n in range(1, len(list_length))]
        output_list = [None] * original_input_length
        # print(len(valid_context_id), valid_context_id[:10], valid_context_id[-10:0])
        # print(original_input_length)
        # print(len(list_question), len(list_answer))
        for n, _id in enumerate(valid_context_id):
            output_list[_id] = [(q, a) for q, a in zip(list_question[n], list_answer[n])]
        return output_list[0] if single_input else output_list

    def generate_a(self,
                   context: str or List,
                   batch_size: int = None,
                   num_beams: int = 4,
                   cache_path: str = None,
                   sentence_level: bool = False,
                   num_questions: int = None):
        """Trich xuat dap an tu context.

        @param context: Van ban dau vao.
        @param batch_size: Batch size.
        @param num_beams: So beam search.
        @param cache_path: Duong dan cache feature da encode.
        @param sentence_level: Bat prediction theo tung cau.
        @param num_questions: So dap an toi da (cho backend spaCy).
        @return: Danh sach dap an cho moi context.
        """
        logging.info(f'running model for `answer_extraction`')
        if self.answer_model_type == 'spacy':
            num_questions = 10 if num_questions is None else num_questions
            if type(context) is str:
                return self.spacy_module.keyword(context, num_questions)
            else:
                return [self.spacy_module.keyword(c, num_questions) for c in context]
        single_input = type(context) is str
        context = [context] if single_input else context
        list_sentences = [self.spacy_module.sentence(c) for c in context]  # Tach context thanh danh sach cau.
        list_inputs = [[c] * len(s) for c, s in zip(context, list_sentences)]
        list_length = [0] + np.cumsum([len(s) for s in list_sentences]).tolist()
        if sentence_level:
            list_inputs = list_sentences
        # Flatten de dua vao batch generate.
        flat_sentences = list(chain(*list_sentences))
        flat_inputs = list(chain(*list_inputs))
        if self.answer_model_type == 'multitask':
            answer = self.generate_prediction(
                flat_inputs,  # list_input,
                highlights=flat_sentences,  # highlights=list_sentence,
                prefix_type='ae' if self.add_prefix else None,
                cache_path=cache_path,
                num_beams=num_beams,
                batch_size=batch_size
            )
        elif self.answer_model_type == 'pipeline':
            answer = self.generate_prediction(
                flat_inputs,  # list_input,
                highlights=flat_sentences,  # highlights=list_sentence,
                prefix_type='ae' if self.add_prefix_ae else None,
                cache_path=cache_path,
                num_beams=num_beams,
                batch_size=batch_size,
                switch_to_model_ae=True
            )
        else:
            raise ValueError(f"unknown answer model type: {self.answer_model_type}")
        # Khoi phuc lai cau truc nested theo context ban dau.
        answer = [clean(a) for a in answer]
        list_answer = [answer[list_length[n - 1]:list_length[n]] for n in range(1, len(list_length))]
        list_answer = [[a for a, c in zip(a_sent, c_sent) if a is not None and a in c]
                       for a_sent, c_sent in zip(list_answer, list_inputs)]
        list_answer = [None if len(a) == 0 else a for a in list_answer]
        if not self.drop_answer_error_text:
            if any(a is None for a in list_answer):
                raise AnswerNotFoundError([context[n] for n, a in enumerate(list_answer) if a is None][0])
        return list_answer[0] if single_input else list_answer

    def generate_q(self,
                   list_context: str or List,
                   list_answer: List = None,
                   batch_size: int = None,
                   num_beams: int = 4,
                   cache_path: str = None,
                   sentence_level: bool = False):
        """Sinh cau hoi tu context va answer (answer duoc highlight bang `<hl>`).

        @param list_context: Context dau vao.
        @param list_answer: Danh sach answer can highlight trong context.
        @param batch_size: Batch size.
        @param num_beams: So beam search.
        @param cache_path: Duong dan cache feature da encode.
        @param sentence_level: Bat prediction theo cau de giam do phuc tap.
        @return: Cau hoi sinh ra.
        """
        assert self.is_qg, "model is not fine-tuned for QG"
        if list_answer is not None:
            assert type(list_context) is type(list_answer), f"{type(list_context)} != {type(list_answer)}"
        single_input = False
        if type(list_context) is str:
            list_context = [list_context]
            list_answer = [list_answer] if list_answer is not None else None
            single_input = True
        output = self.generate_prediction(
            list_context,
            highlights=list_answer,
            prefix_type='qg' if self.add_prefix else None,
            cache_path=cache_path,
            num_beams=num_beams,
            batch_size=batch_size,
            sentence_level=sentence_level
        )
        if single_input:
            return output[0]
        return output

    def answer_q(self,
                 list_context: str or List,
                 list_question: str or List,
                 batch_size: int = None,
                 num_beams: int = 4,
                 cache_path: str = None):
        logging.info(f'running model for `question_answering`')
        assert self.is_qa, "model is not fine-tuned for QA"
        assert type(list_context) is type(list_question), "invalid input"
        single_input = type(list_context) is str
        list_context = [list_context] if single_input else list_context
        list_question = [list_question] if single_input else list_question
        assert len(list_context) == len(list_question), f"invalid input: {len(list_context)} != {len(list_question)}"
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
        """Ham generate tong quat cho QG/AE/QA.

        @param inputs: Danh sach input text.
        @param highlights: Danh sach span can highlight bang `<hl>`.
        @param batch_size: Batch size.
        @param num_beams: So beam search.
        @param cache_path: Duong dan cache feature da encode.
        @param prefix_type: Prefix tac vu (`qg`, `ae`, `qag`, `qa`).
        @return: Danh sach chuoi da generate.
        """
        self.eval()
        if switch_to_model_ae:
            assert self.model_ae is not None and self.tokenizer_ae is not None
            model = self.model_ae
            tokenizer = self.tokenizer_ae
            max_length_output = self.max_length_output_ae
        else:
            model = self.model
            tokenizer = self.tokenizer
            max_length_output = self.max_length_output

        if sentence_level:
            assert highlights is not None, '`sentence_level` can tham so `highlights` de xac dinh cau chua answer.'
            assert len(highlights) == len(inputs), str([len(highlights), len(inputs)])
            list_sentence = []
            for context, answer in zip(inputs, highlights):
                s = [sentence for sentence in self.spacy_module.sentence(context) if answer in sentence]
                list_sentence.append(s[0] if len(s) != 0 else context)
            inputs = list_sentence

        assert type(inputs) is list, inputs
        encode_list = self.text_to_encode(
            inputs,
            highlights=highlights,
            prefix_type=prefix_type,
            cache_path=cache_path,
            switch_to_model_ae=switch_to_model_ae
        )
        loader = self.get_data_loader(encode_list, batch_size=batch_size)
        outputs = []
        for encode in loader:
            with torch.no_grad():
                if 'labels' in encode:
                    encode.pop('labels')
                encode = {k: v.to(self.device) for k, v in encode.items()}
                encode['max_length'] = max_length_output
                encode['num_beams'] = num_beams
                tensor = model.module.generate(**encode) if self.parallel else model.generate(**encode)
                outputs += tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return outputs

    def encode_to_loss(self, encode: Dict):
        """Tinh loss tu feature da encode (dung cho fine-tuning).

        @param encode: Dict feature da tokenize (co `labels`).
        @return: Gia tri loss scalar.
        """
        assert 'labels' in encode
        output = self.model(**{k: v.to(self.device) for k, v in encode.items()})
        if self.label_smoothing is None or self.label_smoothing == 0.0:
            return output['loss'].mean() if self.parallel else output['loss']
        else:
            return label_smoothed_loss(output['logits'], encode['labels'].to(self.device), self.label_smoothing)

    def text_to_encode(self,
                       inputs,
                       outputs: List = None,
                       highlights: List = None,
                       prefix_type: str = None,
                       cache_path: str = None,
                       switch_to_model_ae: bool = False):
        """Chuyen text dau vao/dau ra thanh feature tokenized.

        @param inputs: Danh sach input.
        @param outputs: Danh sach output tuong ung (neu co).
        @param highlights: Danh sach span can danh dau `<hl>`.
        @param prefix_type: Prefix tac vu.
        @param cache_path: Duong dan cache feature trung gian.
        @return: Danh sach feature da encode.
        """
        if cache_path is not None and os.path.exists(cache_path):
            logging.info(f'loading preprocessed feature from {cache_path}')
            return pickle_load(cache_path)
        outputs = [None] * len(inputs) if outputs is None else outputs
        highlights = [None] * len(inputs) if highlights is None else highlights
        assert len(outputs) == len(inputs) == len(highlights), str([len(outputs), len(inputs), len(highlights)])
        data = list(zip(inputs, outputs, highlights))
        # Chon cach xu ly: song song hoac don luong.
        config = {'tokenizer': self.tokenizer, 'max_length': self.max_length, 'prefix_type': prefix_type,
                  'max_length_output': self.max_length_output, 'drop_overflow_error_text': self.drop_overflow_error_text,
                  'skip_overflow_error': self.skip_overflow_error, 'drop_highlight_error_text': self.drop_highlight_error_text,
                  'padding': False if len(data) == 1 else True}
        if switch_to_model_ae:
            assert self.model_ae is not None and self.tokenizer_ae is not None
            config['tokenizer'] = self.tokenizer_ae
            config['max_length'] = self.max_length_ae
            config['max_length_output'] = self.max_length_output_ae

        logging.info(f'encode all the data       : {len(data)}')
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if PARALLEL_PROCESSING:
            pool = Pool()
            out = pool.map(EncodePlus(**config), data)
            pool.close()
            out = list(filter(None, out))  # Loai bo cac mau bi drop (overlength/highlight loi).
        else:
            f = EncodePlus(**config)
            out = []
            files = []
            for i in tqdm(data):
                e = f(i)
                if e is not None:  # Chi giu cac mau hop le.
                    out.append(e)
                if len(out) > 40000 and cache_path is not None:
                    pickle_save(out, f'{cache_path}.tmp{len(files)}')
                    files.append(f'{cache_path}.tmp{len(files)}')
                    out = []
            if len(out) > 0 and cache_path is not None:
                pickle_save(out, f'{cache_path}.tmp{len(files)}')
                files.append(f'{cache_path}.tmp{len(files)}')
            if len(files) > 0:
                out = list(chain(*[pickle_load(i) for i in files]))
        logging.info(f'after remove the overflow : {len(out)}')
        # Luu cache de tai su dung o lan chay sau.
        if cache_path is not None:
            pickle_save(out, cache_path)
            logging.info(f'preprocessed feature is saved at {cache_path}')
        return out

    def save(self, save_dir):
        """Luu model va tokenizer ra thu muc.

        @param save_dir: Thu muc dich.
        """

        def model_state(model):
            if self.parallel:
                return model.module
            return model

        logging.info('saving model')
        model_state(self.model).config.update({'add_prefix': self.add_prefix})
        model_state(self.model).save_pretrained(save_dir)
        logging.info('saving tokenizer')
        self.tokenizer.save_pretrained(save_dir)

    @staticmethod
    def get_data_loader(encode_list, batch_size: int = None, shuffle: bool = False, drop_last: bool = False):
        """Tao DataLoader tu danh sach feature da encode.

        @param encode_list: Danh sach feature da encode.
        @param batch_size: Batch size.
        @param shuffle: Tron du lieu truoc moi epoch.
        @param drop_last: Bo batch cuoi neu khong du so luong.
        @return: torch.utils.data.DataLoader
        """
        batch_size = len(encode_list) if batch_size is None else batch_size
        params = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=NUM_WORKERS)
        return torch.utils.data.DataLoader(Dataset(encode_list), **params)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
"""train.py - CLI Training Script cho ViT5 Question Generation Model
────────────────────────────────────────────────────────────────
Script này cung cấp CLI interface để fine-tune ViT5 model cho:
- Question Generation (QG): Sinh câu hỏi từ đoạn văn + đáp án
- Answer Extraction (AE): Trích xuất đáp án từ đoạn văn + câu hỏi
- Multitask Learning: Train cả 2 tasks cùng lúc với prefix discrimination

Cách chạy:
    python train.py fine_tuning \
        --model='VietAI/vit5-base' \
        --dataset_path='shnl/qg-example' \
        --input_types='["paragraph_answer","paragraph_sentence"]' \
        --output_types='["question","answer"]' \
        --prefix_types='["qg","ae"]' \
        --epoch=10 \
        --batch=4

Ví dụ đơn giản:
    python train.py fine_tuning --model='VietAI/vit5-base'
"""

import fire # type: ignore
from typing import List
import os
from plms.trainer import Trainer

class FineTuning:
    """Wrapper class cho các chức năng training.
    
    Cung cấp CLI methods cho:
    - fine_tuning: Supervised fine-tuning trên QG/AE tasks
    - inst_tuning: Instruction tuning (coming soon)
    - alpaca: Alpaca-style training (coming soon)
    """
    
    def __init__(self):
        """Khởi tạo FineTuning class (không cần state)."""
        pass

    def fine_tuning(
            self,
            checkpoint_dir: str = './cp',
            dataset_path: str = 'shnl/qg-example',
            dataset_name: str = 'default',
            input_types: List or str = ['paragraph_answer', 'paragraph_sentence'], # type: ignore
            output_types: List or str = ['question', 'answer'], # type: ignore
            prefix_types: List or str = ['qg','ae'], # type: ignore
            model: str = '',
            max_length: int = 512,
            max_length_output: int = 256,
            epoch: int = 10,
            batch: int = 4,
            lr: float = 1e-4,
            fp16: bool = False,
            random_seed: int = 42,
            gradient_accumulation_steps: int = 4,
            label_smoothing: float = None,
            disable_log: bool = False,
            config_file: str = 'trainer_config.json',
            use_auth_token: bool = False,
            torch_dtype=None,
            device_map: str = None,
            low_cpu_mem_usage: bool = False
    ):
        """Fine-tune ViT5 model cho Question Generation và Answer Extraction.
        
        Method này:
        1. Load dataset từ HuggingFace hoặc local
        2. Tokenize data với prefix discrimination (qg/ae)
        3. Khởi tạo Trainer với HuggingFace Seq2SeqTrainer
        4. Train model với multitask learning
        5. Lưu checkpoint mỗi epoch vào checkpoint_dir
        
        Args:
            checkpoint_dir: Thư mục lưu model checkpoints
            dataset_path: Path đến dataset (HF hub hoặc local)
            dataset_name: Config name của dataset
            input_types: List input columns cho mỗi task [QG_input, AE_input]
            output_types: List output columns cho mỗi task [QG_output, AE_output]
            prefix_types: List prefix cho multitask ['qg', 'ae']
            model: Model name từ HuggingFace Hub (VD: 'VietAI/vit5-base')
            max_length: Max length của input sequence (tokens)
            max_length_output: Max length của output sequence (tokens)
            epoch: Số epoch training
            batch: Batch size per device
            lr: Learning rate cho AdamW optimizer
            fp16: Enable mixed precision training (FP16)
            random_seed: Random seed
            gradient_accumulation_steps: Accumulate gradients over N steps
            label_smoothing: Label smoothing regularization
            disable_log: Disable wandb logging
            config_file: Config file path
            use_auth_token: Use HF auth token
            torch_dtype: Torch dtype cho model
            device_map: Device mapping strategy
            low_cpu_mem_usage: Low CPU mem mode
        
        Raises:
            AssertionError: Nếu không cung cấp --model parameter
        
        Example:
            python train.py fine_tuning \
                --model='VietAI/vit5-base' \
                --dataset_path='shnl/qg-example' \
                --epoch=10 \
                --batch=4 \
                --lr=1e-4
        """
        # In thông tin training parameters (chỉ trên process chính nếu multi-GPU)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"Training model with params:\n"
                f"checkpoint_dir = {checkpoint_dir}\n"
                f"dataset_path = {dataset_path}\n"
                f"dataset_name = {dataset_name}\n"
                f"input_types = {input_types}\n"
                f"output_types = {output_types}\n"
                f"prefix_types = {prefix_types}\n"
                f"model = {model}\n"
                f"max_length = {max_length}\n"
                f"max_length_output = {max_length_output}\n"
                f"epoch = {epoch}\n"
                f"batch = {batch}\n"
                f"lr = {lr}\n"
                f"fp16 = {fp16}\n"
                f"random_seed = {random_seed}\n"
                f"gradient_accumulation_steps = {gradient_accumulation_steps}\n"
                f"label_smoothing = {label_smoothing}\n"
                f"disable_log = {disable_log}\n"
                f"config_file = {config_file}\n"
                f"use_auth_token = {use_auth_token}\n"
                f"torch_dtype = {torch_dtype}\n"
                f"device_map = {device_map}\n"
                f"low_cpu_mem_usage = {low_cpu_mem_usage}\n"
            )
        
        # Kiểm tra bắt buộc: phải có model name
        assert (
            model
        ), "Please specify a --model, e.g. --model='VietAI/vit5-base'"
        
        # Khởi tạo Trainer với config
        trainer = Trainer(
            checkpoint_dir = checkpoint_dir,
            dataset_path = dataset_path,
            dataset_name = dataset_name,
            input_types = input_types,
            output_types = output_types,
            prefix_types = prefix_types,
            model = model,
            max_length = max_length,
            max_length_output = max_length_output,
            epoch = epoch,
            batch = batch,
            lr = lr,
            fp16 = fp16,
            random_seed = random_seed,
            gradient_accumulation_steps = gradient_accumulation_steps,
            label_smoothing = label_smoothing,
            disable_log = disable_log,
            config_file = config_file,
            use_auth_token = use_auth_token,
            torch_dtype = torch_dtype,
            device_map = device_map,
            low_cpu_mem_usage = low_cpu_mem_usage
        )
        
        # Bắt đầu training loop
        # - Load và tokenize dataset
        # - Setup optimizer, scheduler
        # - Training với evaluation mỗi epoch
        # - Lưu checkpoint sau mỗi epoch
        trainer.train()

    def inst_tuning(self):
        """Instruction tuning method (chưa implement).
        
        Dự kiến: Training theo phong cách instruction-following
        với format: "Instruction: ... Input: ... Output: ..."
        
        Returns:
            str: Placeholder message
        """
        return 'coming soon'

    def alpaca(self):
        """Alpaca-style training method (chưa implement).
        
        Dự kiến: Training theo Stanford Alpaca format
        với self-instruct data generation.
        
        Returns:
            str: Placeholder message
        """
        return 'coming soon'


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Sử dụng Fire để tự động tạo CLI từ class methods
    # Usage: python train.py <method_name> --param=value
    # VD: python train.py fine_tuning --model='VietAI/vit5-base'
    fire.Fire(FineTuning)
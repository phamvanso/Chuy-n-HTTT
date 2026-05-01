""" Training model. """
import os
import json
import logging
import shutil
import random
from os.path import join as pj
from glob import glob
from typing import List
import torch
from tqdm import tqdm

from .language_model import TransformersQG  # Model chính
from .data import get_dataset, DEFAULT_CACHE_DIR  # Load dataset

__all__ = ('to_list', 'Trainer')

# Cờ để đặt optimizer lên CPU (tiết kiệm VRAM khi train model lớn)
OPTIMIZER_ON_CPU = bool(int(os.getenv('OPTIMIZER_ON_CPU', '0')))

# ============================================================================
# HÀM TIỆN ÍCH
# ============================================================================

def to_list(_val, sorting=True):
    """Chuyển giá trị thành list và sắp xếp nếu cần.
    
    Hàm này đảm bảo output luôn là list, và có thể sắp xếp giảm dần.
    Hữu ích khi cần xử lý cả single value và list một cách thống nhất.
    
    Args:
        _val: Giá trị đầu vào (có thể là single value hoặc list)
        sorting: Có sắp xếp giảm dần hay không
        
    Returns:
        List (đã sắp xếp nếu sorting=True)
        
    Examples:
        >>> to_list(5)
        [5]
        >>> to_list([3, 1, 2])
        [3, 2, 1]
    """
    # Nếu không phải list -> chuyển thành list
    if type(_val) != list:
        return [_val]
    # Sắp xếp giảm dần nếu yêu cầu
    if sorting:
        return sorted(_val, reverse=True)
    return _val


# ============================================================================
# CLASS CONFIG - QUẢN LÝ CHECKPOINT VÀ CÁC THAM SỐ
# ============================================================================

class Config:
    """Quản lý checkpoint và configuration của model.
    
    Class này:
    - Lưu/load config từ checkpoint
    - Kiểm tra xem đã có checkpoint với config tương tự chưa
    - Tạo thư mục checkpoint mới nếu cần
    - Đồng bộ config với các thuộc tính của object
    """

    def __init__(self, checkpoint_dir: str, config_file: str = 'trainer_config.json', **kwargs):
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir):
            logging.info(f'load config from existing checkpoint at {self.checkpoint_dir}')
            self.config = self.safe_open(pj(self.checkpoint_dir, config_file))
        else:
            logging.info(f'initialize checkpoint at {self.checkpoint_dir}')
            self.config = kwargs
            configs = {i: self.safe_open(i) for i in glob(pj(os.path.dirname(self.checkpoint_dir), '*', config_file))}
            configs = list(filter(lambda x: x[1] == self.config, configs.items()))
            if len(configs) != 0:
                input(f'\ncheckpoint with same config already exists: {configs[0]}\n enter to overwrite >>>')
                for _p, _ in configs:
                    shutil.rmtree(os.path.dirname(_p))
            self.__initialize_checkpoint(config_file)

        self.__dict__.update(self.config)
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info(f'\t * {k}: {str(v)[:min(100, len(str(v)))]}')

    def __initialize_checkpoint(self, config_file):
        """Tạo thư mục checkpoint và lưu config file.
        
        Args:
            config_file: Tên file config cần lưu
        """
        # Tạo thư mục checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Lưu config vào file JSON
        if not os.path.exists(pj(self.checkpoint_dir, config_file)):
            with open(pj(self.checkpoint_dir, config_file), 'w') as f:
                json.dump(self.config, f)

    @staticmethod
    def safe_open(_file):
        """Mở và đọc file JSON một cách an toàn.
        
        Args:
            _file: Đường dẫn file JSON
            
        Returns:
            Dict đã parse từ JSON
        """
        with open(_file, 'r') as f:
            return json.load(f)


# ============================================================================
# CLASS TRAINER - QUẢN LÝ QUÁ TRÌNH TRAINING
# ============================================================================

class Trainer:
    """Quản lý toàn bộ quá trình training model.
    
    Class này thực hiện:
    - Load/khởi tạo model và optimizer
    - Load dataset và preprocess
    - Training loop với gradient accumulation
    - Lưu checkpoint định kỳ
    - Resume training từ checkpoint
    - Hỗ trợ mixed precision training (FP16)
    """
    def __init__(self,
                 checkpoint_dir: str,
                 dataset_path: str = "shnl/qg-example",
                 dataset_name: str = 'default',
                 input_types: List or str = 'paragraph_answer',
                 output_types: List or str = 'question',
                 prefix_types: List or str = 'qg',
                 model: str = 'VietAI/vit5-base',
                 max_length: int = 512,
                 max_length_output: int = 32,
                 epoch: int = 10,
                 batch: int = 128,
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
                 low_cpu_mem_usage: bool = False):
        """Khởi tạo Trainer.
        
        Args:
            checkpoint_dir: Thư mục lưu checkpoint
            dataset_path: Đường dẫn dataset trên Hugging Face hub
            dataset_name: Tên biến thể của dataset
            input_types: Loại input (paragraph_answer, paragraph_sentence, v.v.)
            output_types: Loại output (question, answer, v.v.)
            prefix_types: Task prefix ('qg', 'ae', 'qa', 'qag')
            model: Tên model trên Hugging Face hub
            max_length: Độ dài tối đa input (tokens)
            max_length_output: Độ dài tối đa output (tokens)
            epoch: Số epoch training
            batch: Batch size
            lr: Learning rate
            fp16: Sử dụng mixed precision training (FP16) hay không
            random_seed: Seed cho reproducibility
            gradient_accumulation_steps: Số step accumulate gradient trước khi update
            label_smoothing: Hệ số label smoothing (0.0-1.0)
            disable_log: Không ghi log vào file
            config_file: Tên file config
            use_auth_token: Token Hugging Face
            torch_dtype: Kiểu dữ liệu tensor
            device_map: Sơ đồ phân bổ model lên GPU
            low_cpu_mem_usage: Giảm bộ nhớ CPU khi load model
        """
        logging.info('initialize model trainer')
        self.use_auth_token = use_auth_token
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        
        # Bước 1: Khởi tạo Config object (quản lý checkpoint và hyperparameters)
        self.config = Config(
            config_file=config_file, checkpoint_dir=checkpoint_dir, dataset_path=dataset_path, dataset_name=dataset_name,
            input_types=input_types, output_types=output_types, prefix_types=prefix_types, model=model,
            max_length=max_length, max_length_output=max_length_output, epoch=epoch, batch=batch, lr=lr, fp16=fp16,
            random_seed=random_seed, gradient_accumulation_steps=gradient_accumulation_steps,
            label_smoothing=label_smoothing)

        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        # Bước 3: Thiết lập file logging (ghi log vào file)
        if not disable_log:
            logger = logging.getLogger()
            file_handler = logging.FileHandler(pj(self.config.checkpoint_dir, 'training.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
            logger.addHandler(file_handler)

        # Bước 4: Load model (từ checkpoint hoặc khởi tạo mới)
        add_prefix = False if self.config.prefix_types is None else True
        ckpts = glob(pj(self.config.checkpoint_dir, 'epoch_*'))
        ckpts = [i for i in ckpts if os.path.exists(
            pj(self.config.checkpoint_dir, 'optimizers', f"optimizer.{i.split('epoch_')[-1]}.pt"))]
        flag = False  # Flag đánh dấu đã load checkpoint thành công chưa
        if len(ckpts):  # Nếu có checkpoint
            # Sắp xếp giảm dần để load epoch mới nhất trước
            epochs = sorted([int(i.split('epoch_')[-1]) for i in ckpts], reverse=True)
            
            # Thử load từng checkpoint (từ mới nhất đến cũ nhất)
            for epoch in epochs:
                try:
                    path = pj(self.config.checkpoint_dir, f"epoch_{epoch}")
                    logging.info(f'load checkpoint from {path}')
                    
                    # Load model từ checkpoint
                    self.model = TransformersQG(
                        model=path, max_length=self.config.max_length, max_length_output=self.config.max_length_output,
                        label_smoothing=self.config.label_smoothing, add_prefix=add_prefix,
                        drop_overflow_error_text=True, use_auth_token=self.use_auth_token, device_map=self.device_map,
                        low_cpu_mem_usage=self.low_cpu_mem_usage, torch_dtype=self.torch_dtype)
                    
                    # Load optimizer state
                    self.optimizer = self.setup_optimizer(epoch)
                    self.current_epoch = epoch
                    
                    # Kiểm tra xem đã train xong chưa
                    assert self.current_epoch <= self.config.epoch, 'model training is done'
                    flag = True
                except Exception:
                    # Nếu checkpoint bị lỗi -> thử checkpoint tiếp theo
                    logging.exception(f'error at loading checkpoint {ckpts}')
                
                # Đã load thành công -> thoát vòng lặp
                if flag:
                    break
        
        # Nếu không load được checkpoint nào -> khởi tạo model mới
        if not flag:
            logging.info(f'initialize checkpoint with {self.config.model}')
            self.model = TransformersQG(
                model=self.config.model, max_length=self.config.max_length,
                max_length_output=self.config.max_length_output, add_prefix=add_prefix,
                drop_overflow_error_text=True, use_auth_token=self.use_auth_token,
                device_map=self.device_map, low_cpu_mem_usage=self.low_cpu_mem_usage, torch_dtype=self.torch_dtype)
            self.optimizer = self.setup_optimizer()
            self.current_epoch = 0  # Bắt đầu từ epoch 0
        
        # Bước 5: Thiết lập GradScaler cho mixed precision training (FP16)
        # Scaler tự động scale gradient để tránh underflow khi dùng FP16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)

        # Bước 6: Chuẩn bị đường dẫn cache cho encoded features
        input_types = to_list(self.config.input_types, sorting=False)
        output_types = to_list(self.config.output_types, sorting=False)
        assert len(input_types) == len(output_types)
        if prefix_types is None:
            prefix_types = [None] * len(input_types)
        else:
            prefix_types = to_list(self.config.prefix_types, sorting=False)
        prefix = pj(
            DEFAULT_CACHE_DIR,
            "encoded_feature"
            f"{self.config.dataset_path}{'.' + self.config.dataset_name if self.config.dataset_name != 'default' else ''}",
            f"{self.config.model}.{self.config.max_length}.{self.config.max_length_output}"
        )

        self.data_cache_paths = [[(i, o, p), f'{prefix}.{i}.{o}.train.{p}.pkl']
                                 for i, o, p in zip(input_types, output_types, prefix_types)]

    def setup_optimizer(self, epoch: int = None):
        """Khởi tạo hoặc load optimizer.
        
        Args:
            epoch: Epoch number (nếu có -> load optimizer từ checkpoint)
            
        Returns:
            torch.optim.Optimizer object
        """
        # Khởi tạo AdamW optimizer
        # AdamW: Adam với weight decay được tách riêng (tốt hơn Adam thông thường)
        optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.config.lr)
        
        if epoch is not None:
            # Load optimizer state từ checkpoint (để resume training)
            path = pj(self.config.checkpoint_dir, "optimizers", f'optimizer.{epoch}.pt')
            logging.info(f'load optimizer from {path}')
            
            # Chọn device: CPU hoặc GPU
            # OPTIMIZER_ON_CPU=True: đặt optimizer lên CPU (tiết kiệm VRAM)
            device = 'cpu' if OPTIMIZER_ON_CPU else self.model.device
            logging.info(f'optimizer is loading on {device}')
            
            # Load state dict và đồng bộ vào optimizer
            optimizer_stat = torch.load(path, map_location=torch.device(device))
            optimizer.load_state_dict(optimizer_stat['optimizer_state_dict'])
            del optimizer_stat  # Giải phóng bộ nhớ
        
        return optimizer

    def save(self, current_epoch):
        """Lưu checkpoint (model + optimizer + config).
        
        Checkpoint bao gồm:
        - Model weights và config
        - Tokenizer
        - Optimizer state
        - Training config với epoch hiện tại
        
        Args:
            current_epoch: Epoch hiện tại (đã hoàn thành)
        """
        # Bước 1: Lưu model và tokenizer
        save_dir = pj(self.config.checkpoint_dir, f'epoch_{current_epoch + 1}')
        os.makedirs(save_dir, exist_ok=True)
        logging.info('saving model related files')
        self.model.save(save_dir)
        
        # Lưu config với epoch đã update
        with open(pj(save_dir, 'trainer_config.json'), 'w') as f:
            tmp = self.config.config.copy()
            tmp['epoch'] = current_epoch + 1  # Update epoch đã train
            json.dump(obj=tmp, fp=f)

        # Bước 2: Lưu optimizer state
        save_dir_opt = pj(self.config.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch + 1}.pt')
        os.makedirs(os.path.dirname(save_dir_opt), exist_ok=True)
        logging.info('saving optimizer')
        torch.save({'optimizer_state_dict': self.optimizer.state_dict()}, save_dir_opt)

        # Bước 3: Xóa optimizer file cũ (tiết kiệm dung lượng)
        logging.info('remove old optimizer files')
        path = pj(self.config.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch}.pt')
        if os.path.exists(path):
            os.remove(path)

    def train(self, epoch_save: None or int = 1, interval: int = 25, epoch_partial: int = None):
        """Train model với full training loop.

        Luồng xử lý:
        1. Load và preprocess dataset
        2. Tạo DataLoader với shuffle
        3. Training loop qua các epoch
        4. Lưu checkpoint định kỳ
        5. Log loss và learning rate
        
        Args:
            epoch_save: Lưu model sau mỗi bao nhiêu epoch (None = không lưu giữa chừng)
            interval: Log loss sau mỗi bao nhiêu global step
            epoch_partial: Dừng sớm sau epoch này (để test nhanh)
        """
        # Chuyển model sang training mode
        self.model.train()

        # Kiểm tra xem đã train xong chưa
        if self.current_epoch == self.config.epoch:
            logging.info('training is completed')
            return None

        # Bước 1: Load và preprocess dataset
        logging.info('dataset preprocessing')
        encode_list = []
        
        # Lặp qua từng task (có thể train multitask)
        for (i, o, p), cache_path in self.data_cache_paths:
            # Load dataset từ Hugging Face
            text_input, text_output = get_dataset(
                self.config.dataset_path, 
                self.config.dataset_name, 
                split='train', 
                input_type=i, 
                output_type=o,
                use_auth_token=self.use_auth_token)
            
            # Tokenize và encode (có cache để tăng tốc)
            encode_list += self.model.text_to_encode(text_input, text_output, prefix_type=p, cache_path=cache_path)
        
        # Tạo DataLoader với shuffle và drop_last
        # drop_last=True: bỏ batch cuối cùng nếu không đủ batch_size (tránh ảnh hưởng gradient)
        loader = self.model.get_data_loader(encode_list, batch_size=self.config.batch, shuffle=True, drop_last=True)

        # Bước 2: Training loop
        logging.info('start model training')
        global_step = 0  # Đếm tổng số lần optimizer update (across epochs)
        saved_checkpoints = []  # Danh sách epoch đã lưu
        
        # Bật autocast cho mixed precision (FP16)
        # autocast tự động chuyển một số operation sang FP16 để tăng tốc và tiết kiệm VRAM
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            # Loop qua các epoch
            for e in tqdm(range(self.current_epoch, self.config.epoch)):
                
                # Train 1 epoch
                mean_loss, global_step = self.train_single_epoch(loader, global_step, interval)
                
                # Log kết quả epoch
                logging.info(f"[epoch {e}/{self.config.epoch}] average loss: {round(mean_loss, 3)}, "
                             f"lr: {self.optimizer.param_groups[0]['lr']}")
                
                # Lưu checkpoint định kỳ nếu epoch_save được chỉ định
                if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != 0:
                    #self.save(e)  # Comment out để không lưu giữa chừng
                    saved_checkpoints.append(e)
                
                # Dừng sớm nếu epoch_partial được chỉ định (để test nhanh)
                if epoch_partial is not None and (e + 1) == epoch_partial:
                    break
            
            # Lưu checkpoint cuối cùng sau khi train xong
            self.save(e)
        
        # Đảm bảo checkpoint cuối cùng được lưu
        if e not in saved_checkpoints:
            self.save(e)
        
        logging.info(f'complete training: model ckpt was saved at {self.config.checkpoint_dir}')

    def train_single_epoch(self, data_loader, global_step: int, interval):
        """Train một epoch duy nhất.
        
        Thực hiện gradient accumulation: accumulate gradient qua nhiều batch
        trước khi update optimizer (giúp tăng effective batch size mà không tốn VRAM).
        
        Args:
            data_loader: DataLoader chứa training data
            global_step: Số lần optimizer đã update (qua các epoch)
            interval: Log loss sau mỗi bao nhiêu step
            
        Returns:
            tuple: (average_loss, updated_global_step)
        """
        total_loss = []  # Lưu loss của tất cả batch
        self.optimizer.zero_grad()  # Reset gradient về 0
        
        # Loop qua từng batch
        for n, encode in tqdm(enumerate(data_loader)):
            # Forward pass: tính loss
            loss = self.model.encode_to_loss(encode)
            
            # Backward pass: tính gradient (có scale cho FP16)
            # scaler.scale() phóng to loss trước khi backward để tránh underflow trong FP16
            self.scaler.scale(loss).backward()
            
            # Lưu loss value (chuyển sang CPU trước khi lưu)
            total_loss.append(loss.cpu().item())
            
            # Gradient accumulation: chỉ update optimizer sau mỗi N batch
            if (n + 1) % self.config.gradient_accumulation_steps != 0:
                continue  # Chưa đủ N batch -> skip optimizer update

            # Đã accumulate đủ gradient -> update optimizer
            global_step += 1
            
            # Tính trung bình loss của N batch vừa accumulate
            _total_loss = total_loss[-self.config.gradient_accumulation_steps:]
            inst_loss = sum(_total_loss)/len(_total_loss)

            # Optimizer update
            # 1. Unscale gradient (chia cho scale factor)
            # 2. Clip gradient nếu cần (trong scaler.step)
            # 3. Update weights
            self.scaler.step(self.optimizer)
            
            # Update scale factor cho lần backward tiếp theo
            self.scaler.update()
            
            # Reset gradient về 0 cho accumulation tiếp theo
            self.optimizer.zero_grad()
            
            # Log loss định kỳ
            if global_step % interval == 0:
                logging.info(f"\t * (global step {global_step}: loss: {inst_loss}, "
                             f"lr: {self.optimizer.param_groups[0]['lr']}")
        
        # Đảm bảo gradient được reset sau epoch
        self.optimizer.zero_grad()
        
        # Trả về loss trung bình của epoch và global_step đã update
        return sum(total_loss)/len(total_loss), global_step
# Standard Library
import os
from os.path import join

# Third Party
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch
import torch

# Hugging Face Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    set_seed
)

# PEFT (Parameter-Efficient Fine-Tuning)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

# BitsAndBytes (bnb)
import bitsandbytes as bnb

# Local
from .argument import setting_args

class QloraLLM:
    def __init__(self, cfg):
        self.args, self.training_args = setting_args(cfg)
        self.checkpoint_dir, completed_training = self.get_last_checkpoint(self.args.output_dir)
        if completed_training:
            print('Detected that training was already completed!')
        self.model, self.tokenizer = self.get_accelerate_model(self.args, self.checkpoint_dir)
        
        self.model.config.use_cache = False
        print('loaded model')
        set_seed(self.args.seed)
        
    def get_last_checkpoint(self, checkpoint_dir):
        if isdir(checkpoint_dir):
            is_completed = exists(join(checkpoint_dir, 'completed'))
            if is_completed: return None, True # already finished
            max_step = 0
            for filename in os.listdir(checkpoint_dir):
                if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                    max_step = max(max_step, int(filename.replace('checkpoint-', '')))
            if max_step == 0: return None, is_completed # training started, but no checkpoint
            checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
            print(f"Found a previous checkpoint at: {checkpoint_dir}")
            return checkpoint_dir, is_completed # checkpoint found!
        return None, False # first training
        
    def find_all_linear_names(self, args, model):
        cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])


        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def get_accelerate_model(self, args, checkpoint_dir):
        # GPUまたはXPUのデバイス数を取得
        n_gpus = 0
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
        elif is_ipex_available() and torch.xpu.is_available():
            n_gpus = torch.xpu.device_count()

        # メモリ設定
        max_memory = f'{args.max_memory_MB}MB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        device_map = "auto"

        # 分散環境の場合の設定
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}
            max_memory = {'': max_memory[local_rank]}

        # フルファインチューニング時のビット数チェック
        if args.full_finetune:
            assert args.bits in [16, 32], "Full finetuning requires 16-bit or 32-bit precision."

        print(f'Loading base model {args.model_name_or_path}...')
        compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

        # モデルのロード
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=compute_dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token
        )

        # bfloat16 のサポートチェック
        if compute_dtype == torch.float16 and args.bits == 4 and torch.cuda.is_bf16_supported():
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('=' * 80)

        # Intel XPU の場合の dtype 調整
        if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
            compute_dtype = torch.bfloat16
            print('Intel XPU does not support float16 yet, so switching to bfloat16')

        # モデルの並列化設定
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

        # トークナイザのロード
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False,  # Fast tokenizer に問題がある場合は False に設定
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
        )

        # パディングトークンの設定
        if tokenizer.pad_token is None:
            print("Pad token is not set. Adding a pad token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        # LLaMA トークナイザの特殊トークン設定
        if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
            print('Adding special tokens for LLaMA tokenizer.')
            tokenizer.add_special_tokens({
                "eos_token": tokenizer.eos_token or "</s>",
                "bos_token": tokenizer.bos_token or "<s>",
                "unk_token": tokenizer.unk_token or "<unk>",
            })
            model.resize_token_embeddings(len(tokenizer))

        # LoRA モジュールの設定
        if not args.full_finetune:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

            if checkpoint_dir is not None:
                print("Loading adapters from checkpoint.")
                model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
            else:
                print(f'Adding LoRA modules...')
                modules = self.find_all_linear_names(args, model)
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)

        # モジュールのデータ型を調整
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if args.bf16:
                    module.to(torch.bfloat16)
            if 'norm' in name:
                module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight') and args.bf16 and module.weight.dtype == torch.float32:
                    module.to(torch.bfloat16)

        return model, tokenizer
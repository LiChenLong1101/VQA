#!/usr/bin/env python3
"""
A-OK-VQA官方风格训练脚本
使用官方数据加载器和评估格式
"""

import os
import torch
import argparse
import pickle
import hashlib
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, 
    TrainingArguments, 
    Trainer,
    set_seed as transformers_set_seed,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import default_collate
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_seed(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers_set_seed(seed)
    print(f"✅ 随机种子已设置为: {seed}")

# A-OK-VQA官方配置
class AOKVQAOfficialConfig:
    """A-OK-VQA官方风格配置"""
    MODEL_NAME = "/mnt/e/deepseek/QwenQwen2.5-VL-3B-Instruct"
    
    # 数据路径配置 - 使用官方路径结构
    AOKVQA_DIR = "/mnt/e/prophet/datasets/aokvqa"
    COCO_DIR = "/mnt/e/prophet/datasets/coco2017"
    
    # 训练参数
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 1
    MAX_GRAD_NORM = 0.5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.05
    
    OUTPUT_DIR = "./3B-aokvqa-official-ckpt"
    MAX_LENGTH = 1024
    
    # 缓存配置
    CACHE_DIR = "./cache"
    CACHE_PROCESSED = True

def get_coco_path(split, image_id, coco_dir):
    """获取COCO图像路径 - 官方函数"""
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012d}.jpg")

def load_aokvqa_data(split):
    """加载A-OK-VQA数据 - 使用HuggingFace datasets格式"""
    print(f"🔄 加载A-OK-VQA {split}数据...")
    
    # 构建数据文件路径
    data_file = os.path.join(AOKVQAOfficialConfig.AOKVQA_DIR, f"aokvqa_v1p0_{split}.json")
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    # 使用HuggingFace datasets加载，节省内存
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=data_file)["train"]
    
    print(f"✅ 加载了 {len(dataset)} 个 {split} 样本")
    return dataset

def preprocess_aokvqa_official(example, processor, split):
    """A-OK-VQA官方风格预处理函数"""
    try:
        question = example["question"]
        question_id = example["question_id"]
        image_id = example["image_id"]
        
        # 获取图像路径
        image_path = get_coco_path(split, image_id, AOKVQAOfficialConfig.COCO_DIR)
        
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            return None
        
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            return None
        
        # 处理答案 - 与测试集格式保持一致
        if split == "test":
            # 测试集：只有选择题选项，没有正确答案
            answer = "Unknown"  # 测试集不用于训练
            choices = example.get("choices", [])
            if choices:
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                enhanced_question = f"{question}\n\n选项:\n{choices_text}"
            else:
                enhanced_question = question
        else:
            # 训练/验证集：优先使用选择题格式，与测试集保持一致
            if example.get("choices") and example.get("correct_choice_idx", -1) >= 0:
                # 优先使用选择题格式，与测试集保持一致
                choices = example["choices"]
                correct_idx = example["correct_choice_idx"]
                if 0 <= correct_idx < len(choices):
                    answer = f"{chr(65 + correct_idx)}. {choices[correct_idx]}"
                else:
                    answer = "Unknown"
            elif example.get("direct_answers") and len(example["direct_answers"]) > 0:
                # 备选：使用直接答案
                answer = example["direct_answers"][0]
            else:
                answer = "Unknown"
            
            # 构建提示词
            choices = example.get("choices", [])
            if choices:
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                enhanced_question = f"{question}\n\n选项:\n{choices_text}"
            else:
                enhanced_question = question
        
        # 构建对话格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": enhanced_question}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": answer}]
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # 提取tensors
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = input_ids.clone()
        
        # Label masking
        assistant_start = "<|im_start|>assistant"
        start_token_ids = processor.tokenizer(assistant_start, add_special_tokens=False)["input_ids"]
        
        def find_sublist(lst, sublst):
            for i in range(len(lst) - len(sublst) + 1):
                if lst[i:i + len(sublst)] == sublst:
                    return i + len(sublst)
            return -1
        
        start_pos = find_sublist(input_ids.tolist(), start_token_ids)
        if start_pos == -1:
            start_pos = len(input_ids) // 2
        
        labels[:start_pos] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_id": question_id,
            "image_id": image_id,
        }
        
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"][0]
        
        # 添加调试信息
        result['text'] = enhanced_question
        result['answer'] = answer
        result['split'] = split
        
        return result
        
    except Exception as e:
        print(f"Error preprocessing A-OK-VQA official example: {e}")
        import traceback
        traceback.print_exc()
        return None

def multimodal_collate_fn(batch):
    """数据整理函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    max_seq_len = 0
    max_height, max_width = 0, 0
    pixel_values_list = []
    max_allowed_length = AOKVQAOfficialConfig.MAX_LENGTH
    
    for item in batch:
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in item:
                tensor = item[key]
                if isinstance(tensor, list):
                    tensor = torch.tensor(tensor, dtype=torch.long)
                if isinstance(tensor, torch.Tensor):
                    seq_len = min(tensor.size(-1), max_allowed_length)
                    max_seq_len = max(max_seq_len, seq_len)
        
        pv = item.get("pixel_values", None)
        if pv is not None:
            if isinstance(pv, list):
                for p in pv:
                    if isinstance(p, torch.Tensor) and p.dim() == 3:
                        max_height = max(max_height, p.size(1))
                        max_width = max(max_width, p.size(2))
            elif isinstance(pv, torch.Tensor) and pv.dim() == 3:
                max_height = max(max_height, pv.size(1))
                max_width = max(max_width, pv.size(2))
    
    padded_batch = []
    for item in batch:
        padded_item = {}
        
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in item:
                tensor = item[key]
                if isinstance(tensor, list):
                    tensor = torch.tensor(tensor, dtype=torch.long)
                
                if isinstance(tensor, torch.Tensor):
                    if tensor.dim() == 1:
                        if key == "labels":
                            padded = torch.full((max_seq_len,), -100, dtype=tensor.dtype)
                        else:
                            padded = torch.zeros(max_seq_len, dtype=tensor.dtype)
                        
                        seq_len = min(tensor.size(0), max_seq_len)
                        padded[:seq_len] = tensor[:seq_len]
                        padded_item[key] = padded
                    else:
                        padded_item[key] = tensor
                else:
                    padded_item[key] = tensor
        
        pv = item.get("pixel_values", None)
        if pv is not None:
            if isinstance(pv, list):
                for p in pv:
                    if isinstance(p, torch.Tensor) and p.dim() == 3:
                        target = torch.zeros(p.size(0), max_height, max_width, dtype=p.dtype)
                        h, w = min(p.size(1), max_height), min(p.size(2), max_width)
                        target[:, :h, :w] = p[:, :h, :w]
                        pixel_values_list.append(target)
            elif isinstance(pv, torch.Tensor) and pv.dim() == 3:
                target = torch.zeros(pv.size(0), max_height, max_width, dtype=pv.dtype)
                h, w = min(pv.size(1), max_height), min(pv.size(2), max_width)
                target[:, :h, :w] = pv[:, :h, :w]
                pixel_values_list.append(target)
        
        for key in ["text", "answer", "question_id", "image_id", "split"]:
            if key in item:
                padded_item[key] = item[key]
        
        padded_batch.append(padded_item)
    
    try:
        collated_data = default_collate(padded_batch)
    except Exception as e:
        print(f"错误: 数据批处理失败: {e}")
        raise e
    
    if pixel_values_list:
        collated_data["pixel_values"] = torch.stack(pixel_values_list, dim=0)
    
    return collated_data

# 缓存功能
def get_cache_path(cache_type, identifier):
    """获取缓存文件路径"""
    os.makedirs(AOKVQAOfficialConfig.CACHE_DIR, exist_ok=True)
    return os.path.join(AOKVQAOfficialConfig.CACHE_DIR, f"{cache_type}_{identifier}.pkl")

def get_data_hash(data_files):
    """计算数据文件的哈希值"""
    hash_md5 = hashlib.md5()
    for file_path in data_files.values():
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_cached_processed_data(processor_name, data_hash):
    """加载缓存的预处理数据"""
    if not AOKVQAOfficialConfig.CACHE_PROCESSED:
        return None, None
    
    cache_key = f"{processor_name}_{data_hash}"
    train_cache_path = get_cache_path("train_processed", cache_key)
    val_cache_path = get_cache_path("val_processed", cache_key)
    
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print(f"🔄 从缓存加载预处理数据...")
        try:
            with open(train_cache_path, 'rb') as f:
                train_dataset = pickle.load(f)
            with open(val_cache_path, 'rb') as f:
                val_dataset = pickle.load(f)
            print("✅ 预处理数据缓存加载成功")
            return train_dataset, val_dataset
        except Exception as e:
            print(f"⚠️ 预处理数据缓存加载失败: {e}")
            return None, None
    return None, None

def save_cached_processed_data(train_dataset, val_dataset, processor_name, data_hash):
    """保存预处理数据到缓存"""
    if not AOKVQAOfficialConfig.CACHE_PROCESSED:
        return
    
    cache_key = f"{processor_name}_{data_hash}"
    train_cache_path = get_cache_path("train_processed", cache_key)
    val_cache_path = get_cache_path("val_processed", cache_key)
    
    try:
        with open(train_cache_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(val_cache_path, 'wb') as f:
            pickle.dump(val_dataset, f)
        print(f"✅ 预处理数据已缓存到: {AOKVQAOfficialConfig.CACHE_DIR}")
    except Exception as e:
        print(f"⚠️ 预处理数据缓存保存失败: {e}")

# A-OK-VQA官方训练器
class AOKVQAOfficialTrainer(Trainer):
    """A-OK-VQA官方风格训练器"""
    
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.best_loss = float('inf')
        self.patience = 3
        self.patience_counter = 0
        self.step_count = 0
        self.debug_steps = 10  # 输出前10个step的详细信息
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算损失"""
        model.train()
        device = next(model.parameters()).device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
        
        # 在前几个step输出详细信息
        if self.step_count < self.debug_steps:
            self._debug_step_details(inputs)
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 输出损失信息
        if self.step_count < self.debug_steps:
            print(f"🔍 Step {self.step_count + 1} - 损失: {loss.item():.4f}")
        
        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        if num_items_in_batch is not None and num_items_in_batch > 0:
            loss = loss * (inputs["input_ids"].size(0) / num_items_in_batch)
        
        self.step_count += 1
        return (loss, outputs) if return_outputs else loss
    
    def _debug_step_details(self, inputs):
        """输出训练step的详细信息"""
        print(f"\n{'='*80}")
        print(f"🔍 Step {self.step_count + 1} 详细信息")
        print(f"{'='*80}")
        
        # 获取batch中的第一个样本
        batch_size = inputs["input_ids"].size(0)
        print(f"📊 Batch大小: {batch_size}")
        
        # 解码输入文本
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"][0]  # 取第一个样本
            try:
                # 解码完整输入
                full_text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
                print(f"📝 完整输入文本:")
                print(f"   {full_text}")
                
                # 找到assistant开始位置
                assistant_start = "<|im_start|>assistant"
                start_token_ids = self.processor.tokenizer(assistant_start, add_special_tokens=False)["input_ids"]
                
                def find_sublist(lst, sublst):
                    for i in range(len(lst) - len(sublst) + 1):
                        if lst[i:i + len(sublst)] == sublst:
                            return i + len(sublst)
                    return -1
                
                start_pos = find_sublist(input_ids.tolist(), start_token_ids)
                if start_pos != -1:
                    # 分离用户输入和助手回复
                    user_input = input_ids[:start_pos]
                    assistant_output = input_ids[start_pos:]
                    
                    user_text = self.processor.tokenizer.decode(user_input, skip_special_tokens=False)
                    assistant_text = self.processor.tokenizer.decode(assistant_output, skip_special_tokens=False)
                    
                    print(f"\n👤 用户输入:")
                    print(f"   {user_text}")
                    print(f"\n🤖 助手回复:")
                    print(f"   {assistant_text}")
                else:
                    print("⚠️ 未找到assistant开始标记")
                    
            except Exception as e:
                print(f"❌ 解码失败: {e}")
        
        # 显示labels信息
        if "labels" in inputs:
            labels = inputs["labels"][0]
            valid_labels = labels[labels != -100]
            masked_count = (labels == -100).sum().item()
            valid_count = len(valid_labels)
            
            print(f"\n🏷️ Labels信息:")
            print(f"   总长度: {len(labels)}")
            print(f"   被mask的数量: {masked_count}")
            print(f"   计算损失的数量: {valid_count}")
            
            if len(valid_labels) > 0:
                try:
                    valid_text = self.processor.tokenizer.decode(valid_labels, skip_special_tokens=False)
                    print(f"   有效labels文本: {valid_text}")
                except:
                    print(f"   有效labels tokens: {valid_labels.tolist()[:10]}...")
        
        # 显示图像信息
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"][0]
            print(f"\n🖼️ 图像信息:")
            print(f"   形状: {pixel_values.shape}")
            print(f"   数据类型: {pixel_values.dtype}")
            print(f"   数值范围: [{pixel_values.min().item():.3f}, {pixel_values.max().item():.3f}]")
        
        # 显示其他调试信息
        if hasattr(inputs, 'get') and 'text' in inputs:
            print(f"\n📋 原始问题: {inputs.get('text', 'N/A')}")
            print(f"📋 原始答案: {inputs.get('answer', 'N/A')}")
        
        print(f"{'='*80}\n")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估回调"""
        if metrics is not None:
            current_loss = metrics.get('eval_loss', float('inf'))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
                print(f"🎉 新的最佳损失: {current_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"📊 当前损失: {current_loss:.4f}, 最佳损失: {self.best_loss:.4f}, 耐心计数: {self.patience_counter}")
            
            if self.patience_counter >= self.patience:
                print(f"🛑 早停触发: 连续 {self.patience} 次评估无改善")
                control.should_training_stop = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """训练日志回调"""
        if logs is not None:
            loss = logs.get('loss', 0.0)
            grad_norm = logs.get('grad_norm', 0.0)
            learning_rate = logs.get('learning_rate', 0.0)
            
            if state.global_step % 10 == 0:
                print(f"🔄 步骤 {state.global_step}:")
                print(f"   损失: {loss:.4f}")
                print(f"   梯度范数: {grad_norm:.4f}")
                print(f"   学习率: {learning_rate:.2e}")

def main():
    """主训练函数 - A-OK-VQA官方风格"""
    print("🚀 开始A-OK-VQA官方风格训练...")
    print("=" * 50)
    
    set_seed(42)
    
    # 检查数据路径
    print("🔄 检查数据路径...")
    if not os.path.exists(AOKVQAOfficialConfig.AOKVQA_DIR):
        print(f"❌ A-OK-VQA数据目录不存在: {AOKVQAOfficialConfig.AOKVQA_DIR}")
        return
    
    if not os.path.exists(AOKVQAOfficialConfig.COCO_DIR):
        print(f"❌ COCO图像目录不存在: {AOKVQAOfficialConfig.COCO_DIR}")
        return
    
    print("✅ 数据路径检查通过")
    
    # 加载处理器
    print("🔄 加载处理器...")
    try:
        processor = AutoProcessor.from_pretrained(AOKVQAOfficialConfig.MODEL_NAME)
        print("✅ 处理器加载成功")
    except Exception as e:
        print(f"❌ 处理器加载失败: {e}")
        return
    
    # 加载基础模型
    print("🔄 加载基础模型...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 使用设备: {device}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            AOKVQAOfficialConfig.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "12.0GB"},
            trust_remote_code=True
        )
        
        has_meta_tensors = any(
            param.device.type == 'meta' 
            for param in model.parameters() 
            if hasattr(param, 'device')
        )
        
        if has_meta_tensors:
            print("🔧 检测到 meta 张量，重新加载模型到目标设备...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                AOKVQAOfficialConfig.MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True
            )
            model = model.to(device)
        
        print("✅ 基础模型加载成功")
    except Exception as e:
        print(f"❌ 基础模型加载失败: {e}")
        return
    
    # 配置LoRA
    print("🔄 配置LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.train()
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✅ 梯度检查点已启用")
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        print("✅ 使用 enable_input_require_grads() 启用输入梯度")
    
    print("📊 LoRA应用后的参数统计:")
    model.print_trainable_parameters()
    
    # 加载A-OK-VQA数据集 - 使用HuggingFace datasets格式
    print("🔄 加载A-OK-VQA数据集...")
    try:
        # 使用HuggingFace datasets加载，节省内存
        from datasets import load_dataset
        
        datasets = load_dataset(
            "json",
            data_files={
                "train": os.path.join(AOKVQAOfficialConfig.AOKVQA_DIR, "aokvqa_v1p0_train.json"),
                "validation": os.path.join(AOKVQAOfficialConfig.AOKVQA_DIR, "aokvqa_v1p0_val.json")
            }
        )
        print("✅ A-OK-VQA数据集加载成功")
    except Exception as e:
        print(f"❌ A-OK-VQA数据集加载失败: {e}")
        return
    
    # 预处理数据集 - 使用缓存避免重复处理
    print("🔄 预处理A-OK-VQA数据集...")
    
    # 检查缓存
    data_hash = get_data_hash({
        "train": os.path.join(AOKVQAOfficialConfig.AOKVQA_DIR, "aokvqa_v1p0_train.json"),
        "val": os.path.join(AOKVQAOfficialConfig.AOKVQA_DIR, "aokvqa_v1p0_val.json")
    })
    processor_name = AOKVQAOfficialConfig.MODEL_NAME.replace("/", "_")
    
    train_dataset, val_dataset = load_cached_processed_data(processor_name, data_hash)
    
    if train_dataset is None or val_dataset is None:
        print("🔄 未找到缓存，开始预处理...")
        
        # 使用HuggingFace datasets的map方法，内存友好
        train_dataset = datasets["train"].map(
            lambda x: preprocess_aokvqa_official(x, processor, "train"),
            remove_columns=datasets["train"].column_names,
            desc="预处理A-OK-VQA训练集"
        )
        val_dataset = datasets["validation"].map(
            lambda x: preprocess_aokvqa_official(x, processor, "val"),
            remove_columns=datasets["validation"].column_names,
            desc="预处理A-OK-VQA验证集"
        )
        
        # 过滤掉None的样本
        train_dataset = train_dataset.filter(lambda x: x is not None)
        val_dataset = val_dataset.filter(lambda x: x is not None)
        
        # 保存缓存
        save_cached_processed_data(train_dataset, val_dataset, processor_name, data_hash)
    else:
        print("✅ 从缓存加载预处理数据")
    
    print(f"✅ A-OK-VQA数据集预处理完成")
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   验证集大小: {len(val_dataset)}")
    
    # 创建输出目录
    os.makedirs(AOKVQAOfficialConfig.OUTPUT_DIR, exist_ok=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=AOKVQAOfficialConfig.OUTPUT_DIR,
        per_device_train_batch_size=AOKVQAOfficialConfig.BATCH_SIZE,
        per_device_eval_batch_size=AOKVQAOfficialConfig.BATCH_SIZE,
        gradient_accumulation_steps=AOKVQAOfficialConfig.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=AOKVQAOfficialConfig.LEARNING_RATE,
        num_train_epochs=AOKVQAOfficialConfig.NUM_EPOCHS,
        max_grad_norm=AOKVQAOfficialConfig.MAX_GRAD_NORM,
        warmup_ratio=AOKVQAOfficialConfig.WARMUP_RATIO,
        weight_decay=AOKVQAOfficialConfig.WEIGHT_DECAY,
        
        use_cpu=not torch.cuda.is_available(),
        
        logging_steps=1,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_drop_last=True,
        
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        
        save_total_limit=3,
        seed=42,
        label_smoothing_factor=0.1,
    )
    
    # 创建训练器
    trainer = AOKVQAOfficialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=multimodal_collate_fn,
        processor=processor,
    )
    
    # 开始训练
    print("🚀 开始A-OK-VQA官方风格训练...")
    print("=" * 50)
    
    try:
        trainer.train()
        print("✅ A-OK-VQA官方风格训练完成!")
        
        print("🔄 保存最终模型...")
        trainer.save_model()
        processor.save_pretrained(AOKVQAOfficialConfig.OUTPUT_DIR)
        print(f"✅ 模型已保存到: {AOKVQAOfficialConfig.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

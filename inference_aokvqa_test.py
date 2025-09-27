#!/usr/bin/env python3
"""
A-OK-VQA测试集推理脚本
专门用于测试集推理，生成官方评估格式的结果
"""

import os
import json
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm

class AOKVQATestConfig:
    """A-OK-VQA测试集推理配置"""
    
    # 模型路径
    MODEL_PATH = "./3B-aokvqa-official-ckpt"
    BASE_MODEL_NAME = "/mnt/e/deepseek/QwenQwen2.5-VL-3B-Instruct"
    
    # 数据路径
    AOKVQA_DATA_DIR = "/mnt/e/prophet/datasets/aokvqa"
    COCO_BASE_DIR = "/mnt/e/prophet/datasets/coco2017"
    
    # 推理参数
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.1
    TOP_P = 0.9
    DO_SAMPLE = True
    
    # 输出路径
    OUTPUT_DIR = "./aokvqa_evaluation_results"
    RESULTS_FILE = "aokvqa_evaluation_predictions.json"

def get_coco_path(image_id, split="test"):
    """获取COCO图像路径"""
    # 根据split参数选择正确的COCO图像目录
    if split == "val":
        coco_images_dir = os.path.join(AOKVQATestConfig.COCO_BASE_DIR, "val2017")
    elif split == "test":
        coco_images_dir = os.path.join(AOKVQATestConfig.COCO_BASE_DIR, "test2017")
    else:
        # 默认使用val2017
        coco_images_dir = os.path.join(AOKVQATestConfig.COCO_BASE_DIR, "val2017")
    
    return os.path.join(coco_images_dir, f"{image_id:012d}.jpg")

def load_model_and_processor(model_path, base_model_name):
    """加载模型和处理器"""
    print(f"🔄 加载模型和处理器...")
    print(f"   模型路径: {model_path}")
    print(f"   基础模型: {base_model_name}")
    
    try:
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        print("✅ 处理器加载成功")
        
        # 加载基础模型
        print("🔄 加载基础模型...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ 基础模型加载成功")
        
        # 检查是否是LoRA适配器
        if os.path.exists(model_path):
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print("🔄 检测到LoRA适配器，正在加载...")
                model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    torch_dtype=torch.bfloat16
                )
                print("✅ LoRA适配器加载成功")
            else:
                # 尝试作为完整模型加载
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✅ 训练好的模型加载成功")
        else:
            print(f"⚠️ 训练好的模型不存在: {model_path}")
            print(f"🔄 使用基础模型: {base_model_name}")
            model = base_model
            print("✅ 使用基础模型")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

def preprocess_test_question(question, choices):
    """预处理测试集问题"""
    if choices and len(choices) > 0:
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        enhanced_question = f"{question}\n\n选项:\n{choices_text}"
    else:
        enhanced_question = question
    
    return enhanced_question

def generate_test_answer(model, processor, image, question, choices, config=None):
    """生成测试集答案"""
    if config is None:
        config = AOKVQATestConfig()
    
    # 预处理问题
    enhanced_question = preprocess_test_question(question, choices)
    
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": enhanced_question}
            ]
        }
    ]
    
    # 应用chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理输入
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )
    
    # 移动到GPU
    device = next(model.parameters()).device
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "do_sample": config.DO_SAMPLE,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    
    # 生成答案
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # 解码输出
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return answer, enhanced_question

def extract_choice_from_answer(answer, choices):
    """从答案中提取选择题选项"""
    import re
    
    answer = answer.strip().lower()
    
    # 1. 首先尝试匹配 A. xxx 格式
    pattern = r'^([A-D])\.?\s*(.*)$'
    match = re.match(pattern, answer)
    
    if match:
        choice_letter = match.group(1).upper()
        choice_text = match.group(2).strip()
        return choice_letter, choice_text
    
    # 2. 尝试匹配单独的字母
    answer_upper = answer.upper()
    if answer_upper in ['A', 'B', 'C', 'D']:
        return answer_upper, answer
    
    # 3. 通过内容匹配选项
    if choices:
        for i, choice in enumerate(choices):
            choice_lower = choice.lower().strip()
            # 完全匹配
            if answer == choice_lower:
                return chr(65 + i), choice
            
            # 智能匹配：处理空格、连字符等变体
            answer_normalized = answer.replace(' ', '').replace('-', '').replace('_', '')
            choice_normalized = choice_lower.replace(' ', '').replace('-', '').replace('_', '')
            
            if answer_normalized == choice_normalized:
                return chr(65 + i), choice
            
            # 包含匹配
            if choice_lower in answer or answer in choice_lower:
                return chr(65 + i), choice
            
            # 标准化后的包含匹配
            if choice_normalized in answer_normalized or answer_normalized in choice_normalized:
                return chr(65 + i), choice
    
    # 4. 尝试从答案中提取选项字母
    if answer_upper.startswith(('A', 'B', 'C', 'D')):
        return answer_upper[0], answer[1:].strip('. ').strip()
    
    return None, answer

def calculate_evaluation_metrics(results, dataset):
    """计算评估指标"""
    print("\n📊 计算评估指标...")
    
    # 基本统计
    total_samples = len(results)
    valid_choices = sum(1 for r in results if r.get("predicted_choice") is not None)
    choice_extraction_rate = valid_choices / total_samples if total_samples > 0 else 0
    
    # 创建数据集字典用于快速查找
    dataset_dict = {item["question_id"]: item for item in dataset}
    
    # 计算准确率
    correct_answers = 0
    correct_choices = 0
    direct_answers = 0
    choice_answers = 0
    
    # 详细分析
    analysis_results = {
        "total_samples": total_samples,
        "valid_choices": valid_choices,
        "choice_extraction_rate": choice_extraction_rate,
        "correct_answers": 0,
        "correct_choices": 0,
        "direct_answers": 0,
        "choice_answers": 0,
        "accuracy": 0.0,
        "choice_accuracy": 0.0,
        "direct_answer_accuracy": 0.0,
        "choice_answer_accuracy": 0.0,
        "detailed_analysis": []
    }
    
    for result in results:
        question_id = result["question_id"]
        predicted_choice = result.get("predicted_choice")
        predicted_text = result.get("predicted_text", "")
        raw_answer = result.get("raw_answer", "")
        
        # 获取真实答案
        if question_id in dataset_dict:
            ground_truth = dataset_dict[question_id]
            correct_answer = ground_truth.get("direct_answers", [])
            correct_choice = ground_truth.get("choices", [])
            correct_choice_idx = ground_truth.get("correct_choice_idx", -1)
            
            # 判断是否为选择题
            is_choice_question = len(correct_choice) > 0 and correct_choice_idx >= 0
            
            if is_choice_question:
                choice_answers += 1
                # 选择题评估 - 比较选项索引
                if predicted_choice and predicted_choice in ['A', 'B', 'C', 'D']:
                    predicted_choice_idx = ord(predicted_choice) - ord('A')
                    if predicted_choice_idx == correct_choice_idx:
                        correct_choices += 1
            else:
                direct_answers += 1
                # 直接答案评估
                if correct_answer and predicted_text:
                    # 检查预测答案是否在正确答案列表中
                    predicted_lower = predicted_text.lower().strip()
                    correct_lower = [ans.lower().strip() for ans in correct_answer]
                    if predicted_lower in correct_lower:
                        correct_answers += 1
            
            # 计算当前样本是否正确
            current_correct = False
            if is_choice_question:
                if predicted_choice and predicted_choice in ['A', 'B', 'C', 'D']:
                    predicted_choice_idx = ord(predicted_choice) - ord('A')
                    current_correct = (predicted_choice_idx == correct_choice_idx)
            else:
                if correct_answer and predicted_text:
                    predicted_lower = predicted_text.lower().strip()
                    correct_lower = [ans.lower().strip() for ans in correct_answer]
                    current_correct = (predicted_lower in correct_lower)
            
            # 详细分析记录
            analysis_results["detailed_analysis"].append({
                "question_id": question_id,
                "question": result.get("question", ""),
                "is_choice_question": is_choice_question,
                "predicted_choice": predicted_choice,
                "predicted_text": predicted_text,
                "raw_answer": raw_answer,
                "correct_answer": correct_answer if not is_choice_question else correct_choice,
                "correct_choice_idx": correct_choice_idx if is_choice_question else None,
                "is_correct": current_correct
            })
    
    # 计算最终指标
    analysis_results["correct_answers"] = correct_answers
    analysis_results["correct_choices"] = correct_choices
    analysis_results["direct_answers"] = direct_answers
    analysis_results["choice_answers"] = choice_answers
    
    # 计算准确率
    if direct_answers > 0:
        analysis_results["direct_answer_accuracy"] = correct_answers / direct_answers
    if choice_answers > 0:
        analysis_results["choice_answer_accuracy"] = correct_choices / choice_answers
    
    # 总体准确率
    total_correct = correct_answers + correct_choices
    analysis_results["accuracy"] = total_correct / total_samples if total_samples > 0 else 0
    
    return analysis_results

def print_evaluation_report(metrics):
    """打印评估报告"""
    print("\n" + "="*60)
    print("📈 A-OK-VQA 验证集评估报告")
    print("="*60)
    
    print(f"\n📊 基本统计:")
    print(f"   总样本数: {metrics['total_samples']}")
    print(f"   有效选择题提取: {metrics['valid_choices']} ({metrics['choice_extraction_rate']:.2%})")
    print(f"   直接答案问题: {metrics['direct_answers']}")
    print(f"   选择题问题: {metrics['choice_answers']}")
    
    print(f"\n🎯 准确率指标:")
    print(f"   总体准确率: {metrics['accuracy']:.2%}")
    print(f"   直接答案准确率: {metrics['direct_answer_accuracy']:.2%}")
    print(f"   选择题准确率: {metrics['choice_answer_accuracy']:.2%}")
    
    print(f"\n✅ 正确答案统计:")
    print(f"   直接答案正确: {metrics['correct_answers']}/{metrics['direct_answers']}")
    print(f"   选择题正确: {metrics['correct_choices']}/{metrics['choice_answers']}")
    print(f"   总正确数: {metrics['correct_answers'] + metrics['correct_choices']}/{metrics['total_samples']}")
    
    # 显示一些错误案例
    print(f"\n❌ 错误案例分析 (前5个):")
    error_cases = [case for case in metrics['detailed_analysis'] if not case['is_correct']]
    for i, case in enumerate(error_cases[:5]):
        print(f"   案例 {i+1}:")
        print(f"     问题: {case['question'][:80]}...")
        print(f"     预测: {case['predicted_text']}")
        print(f"     正确答案: {case['correct_answer']}")
        print(f"     类型: {'选择题' if case['is_choice_question'] else '直接答案'}")
        print()

def run_test_inference(model, processor, dataset, config, max_samples=None, split="test"):
    """在数据集上运行推理"""
    print(f"🔄 开始{split}集推理，数据集大小: {len(dataset)}")
    if max_samples:
        dataset = dataset[:max_samples]
        print(f"   限制样本数: {len(dataset)}")
    
    results = []
    
    for i, example in enumerate(tqdm(dataset, desc=f"{split}集推理进度")):
        try:
            # 获取图像
            image_id = example["image_id"]
            image_path = get_coco_path(image_id, split)
            
            if not os.path.exists(image_path):
                print(f"⚠️ 图像不存在: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            
            # 获取问题信息
            question = example["question"]
            choices = example.get("choices", [])
            question_id = example.get("question_id", f"q_{i}")
            
            # 生成答案
            answer, enhanced_question = generate_test_answer(
                model, processor, image, question, choices, config
            )
            
            # 提取选择题答案
            choice_letter, choice_text = extract_choice_from_answer(answer, choices)
            
            # 构建结果 - 测试集格式
            result = {
                "question_id": question_id,
                "image_id": image_id,
                "question": question,
                "choices": choices,
                "enhanced_question": enhanced_question,
                "raw_answer": answer,
                "predicted_choice": choice_letter,
                "predicted_text": choice_text,
            }
            
            # 测试集没有正确答案，所以不添加正确性检查
            results.append(result)
            
            # 输出前几个样本的详细信息
            if i < 5:
                print(f"\n📋 测试样本 {i+1}:")
                print(f"   问题: {question}")
                print(f"   选项: {choices}")
                print(f"   原始答案: {answer}")
                print(f"   提取选项: {choice_letter}")
                print(f"   提取文本: {choice_text}")
        
        except Exception as e:
            print(f"❌ 处理测试样本 {i} 失败: {e}")
            continue
    
    return results

def save_test_results(results, output_dir, results_file):
    """保存测试集推理结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, results_file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试集推理结果已保存到: {output_path}")
    
    # 统计信息
    total_samples = len(results)
    valid_choices = sum(1 for r in results if r.get("predicted_choice") is not None)
    
    print(f"📊 测试集推理统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   有效选择题答案: {valid_choices}")
    print(f"   选择题答案率: {valid_choices/total_samples:.4f}")

def create_official_format(results):
    """创建官方评估格式的结果文件"""
    official_results = {}
    
    for result in results:
        question_id = result["question_id"]
        predicted_choice = result.get("predicted_choice")
        predicted_text = result.get("predicted_text", "")
        raw_answer = result.get("raw_answer", "")
        
        # 官方格式需要包含 multiple_choice 和 direct_answer 两个字段
        entry = {}
        
        # Multiple Choice 预测
        if predicted_choice and predicted_choice in ['A', 'B', 'C', 'D']:
            entry["multiple_choice"] = predicted_choice
        else:
            # 如果没有有效预测，使用默认值
            entry["multiple_choice"] = "A"
        
        # Direct Answer 预测
        if predicted_text:
            entry["direct_answer"] = predicted_text
        elif raw_answer:
            entry["direct_answer"] = raw_answer
        else:
            entry["direct_answer"] = "unknown"
        
        official_results[question_id] = entry
    
    return official_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="A-OK-VQA验证集推理和评估")
    parser.add_argument("--model_path", type=str, default=AOKVQATestConfig.MODEL_PATH,
                       help="训练好的模型路径")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大样本数（用于测试）")
    parser.add_argument("--output_dir", type=str, default=AOKVQATestConfig.OUTPUT_DIR,
                       help="输出目录")
    parser.add_argument("--official_format", action="store_true",
                       help="生成官方评估格式的结果文件")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                       help="数据集分割 (val: 验证集, test: 测试集)")
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="是否进行详细评估 (默认开启)")
    
    args = parser.parse_args()
    
    print("🚀 A-OK-VQA测试集推理和评估")
    print("="*60)
    
    # 加载模型
    model, processor = load_model_and_processor(args.model_path, AOKVQATestConfig.BASE_MODEL_NAME)
    
    # 加载数据集
    print(f"\n🔄 加载A-OK-VQA {args.split}集...")
    data_file = os.path.join(AOKVQATestConfig.AOKVQA_DATA_DIR, f"aokvqa_v1p0_{args.split}.json")
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ {args.split}集加载成功，样本数: {len(dataset)}")
    
    # 运行推理
    config = AOKVQATestConfig()
    results = run_test_inference(model, processor, dataset, config, args.max_samples, args.split)
    
    # 保存详细结果
    detailed_file = f"aokvqa_{args.split}_detailed_predictions.json"
    save_test_results(results, args.output_dir, detailed_file)
    
    # 进行详细评估（仅验证集有正确答案）
    if args.evaluate and args.split == "val":
        print(f"\n🔄 进行详细评估...")
        metrics = calculate_evaluation_metrics(results, dataset)
        
        # 打印评估报告
        print_evaluation_report(metrics)
        
        # 保存评估结果
        metrics_file = f"aokvqa_{args.split}_evaluation_metrics.json"
        metrics_path = os.path.join(args.output_dir, metrics_file)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 评估指标已保存到: {metrics_path}")
    elif args.split == "test":
        print(f"\n⚠️ 测试集没有正确答案，无法进行准确率评估")
        print(f"   请使用官方评估脚本评估测试集结果")
    
    # 生成官方格式结果（默认生成）
    print(f"\n🔄 生成官方评估格式...")
    official_results = create_official_format(results)
    
    official_file = f"predictions_{args.split}.json"
    official_path = os.path.join(args.output_dir, official_file)
    
    with open(official_path, 'w', encoding='utf-8') as f:
        json.dump(official_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 官方格式结果已保存到: {official_path}")
    print(f"   格式: {{'question_id': {{'multiple_choice': 'A', 'direct_answer': 'answer'}}}}")
    print(f"   样本: {dict(list(official_results.items())[:2])}")
    
    # 如果用户指定了 --official_format，也生成旧格式
    if args.official_format:
        legacy_file = f"aokvqa_{args.split}_legacy_predictions.json"
        legacy_path = os.path.join(args.output_dir, legacy_file)
        
        # 生成简化的旧格式
        legacy_results = {}
        for question_id, entry in official_results.items():
            legacy_results[question_id] = entry["multiple_choice"]
        
        with open(legacy_path, 'w', encoding='utf-8') as f:
            json.dump(legacy_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 旧格式结果已保存到: {legacy_path}")
    
    print(f"\n🎉 {args.split}集推理完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    if args.split == "val":
        print(f"💡 验证集评估已完成，可查看详细指标")
    else:
        print(f"💡 测试集推理完成，请使用官方评估脚本评估结果")
        print(f"   官方格式文件: {official_file}")
        print(f"   运行官方评估: python run_official_evaluation.py --predictions {official_path}")

if __name__ == "__main__":
    main()

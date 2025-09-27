#!/usr/bin/env python3
"""
A-OK-VQA数据集预处理脚本
将原始A-OK-VQA数据集转换为训练所需的格式
"""

import os
import json
import argparse
from tqdm import tqdm
import random

def process_aokvqa_dataset(input_dir, output_dir, coco_images_dir):
    """
    处理A-OK-VQA数据集
    
    Args:
        input_dir: A-OK-VQA原始数据目录
        output_dir: 输出目录
        coco_images_dir: COCO2017图像目录
    """
    print("🚀 开始处理A-OK-VQA数据集...")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集和验证集
    for split in ["train", "val"]:
        print(f"🔄 处理{split}集...")
        
        # 原始数据文件路径
        if split == "train":
            input_file = os.path.join(input_dir, "aokvqa_v1p0_train.json")
        else:
            input_file = os.path.join(input_dir, "aokvqa_v1p0_val.json")
        
        if not os.path.exists(input_file):
            print(f"❌ 原始数据文件不存在: {input_file}")
            continue
        
        # 输出文件路径
        output_file = os.path.join(output_dir, f"{split}.json")
        
        # 加载原始数据
        print(f"📂 加载原始数据: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"📊 原始数据包含 {len(raw_data)} 个样本")
        
        # 处理数据
        processed_data = []
        valid_count = 0
        invalid_count = 0
        
        for item in tqdm(raw_data, desc=f"处理{split}集"):
            try:
                # 提取基本信息
                question = item.get("question", "")
                image_id = item.get("image_id", "")
                direct_answers = item.get("direct_answers", [])
                multiple_choice_answer = item.get("multiple_choice_answer", "")
                
                # 检查必要字段
                if not question or not image_id or not direct_answers:
                    invalid_count += 1
                    continue
                
                # 构建图像路径 - COCO2017图像文件名格式为 000000299207.jpg
                image_filename = f"{image_id:012d}.jpg"
                val_image_path = os.path.join(coco_images_dir, "val2017", image_filename)
                train_image_path = os.path.join(coco_images_dir, "train2017", image_filename)
                
                # 检查图像是否存在
                image_path = None
                if os.path.exists(val_image_path):
                    image_path = val_image_path
                elif os.path.exists(train_image_path):
                    image_path = train_image_path
                
                if not image_path:
                    invalid_count += 1
                    continue
                
                # 构建处理后的数据格式 - 适配A-OK-VQA格式
                processed_item = {
                    "question": question,
                    "image_id": image_id,
                    "direct_answers": direct_answers,
                    "multiple_choice_answer": multiple_choice_answer,
                    "choices": item.get("choices", []),
                    "correct_choice_idx": item.get("correct_choice_idx", -1),
                    "rationales": item.get("rationales", []),
                    "difficult_direct_answer": item.get("difficult_direct_answer", False),
                    "image_path": image_path,
                    "image_filename": image_filename
                }
                
                processed_data.append(processed_item)
                valid_count += 1
                
            except Exception as e:
                print(f"⚠️ 处理样本时出错: {e}")
                invalid_count += 1
                continue
        
        # 保存处理后的数据
        print(f"💾 保存处理后的数据到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {split}集处理完成:")
        print(f"   有效样本: {valid_count}")
        print(f"   无效样本: {invalid_count}")
        print(f"   总样本: {len(raw_data)}")
        print()
    
    print("🎉 A-OK-VQA数据集处理完成!")
    print("=" * 50)

def verify_processed_data(output_dir, coco_images_dir):
    """验证处理后的数据"""
    print("🔄 验证处理后的数据...")
    
    for split in ["train", "val"]:
        data_file = os.path.join(output_dir, f"{split}.json")
        
        if not os.path.exists(data_file):
            print(f"❌ 处理后的数据文件不存在: {data_file}")
            continue
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 {split}集统计:")
        print(f"   样本数量: {len(data)}")
        
        # 随机检查几个样本
        if len(data) > 0:
            sample_indices = random.sample(range(min(5, len(data))), min(3, len(data)))
            
            for i, idx in enumerate(sample_indices):
                sample = data[idx]
                print(f"   样本 {i+1}:")
                print(f"     问题: {sample.get('question', 'N/A')[:50]}...")
                print(f"     图像ID: {sample.get('image_id', 'N/A')}")
                print(f"     答案: {sample.get('direct_answers', [])}")
                print(f"     图像路径: {sample.get('image_path', 'N/A')}")
                
                # 检查图像是否存在
                image_path = sample.get('image_path', '')
                if image_path and os.path.exists(image_path):
                    print(f"     ✅ 图像文件存在")
                else:
                    print(f"     ❌ 图像文件不存在")
                print()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理A-OK-VQA数据集")
    parser.add_argument("--input_dir", type=str, default="/mnt/e/prophet/datasets/aokvqa",
                       help="A-OK-VQA原始数据目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/e/prophet/datasets/aokvqa_processed",
                       help="处理后数据输出目录")
    parser.add_argument("--coco_images_dir", type=str, default="/mnt/e/prophet/datasets/coco2017",
                       help="COCO2017图像目录")
    parser.add_argument("--verify_only", action="store_true",
                       help="仅验证已处理的数据")
    
    args = parser.parse_args()
    
    print("🔧 A-OK-VQA数据集预处理工具")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"COCO图像目录: {args.coco_images_dir}")
    print()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    if not os.path.exists(args.coco_images_dir):
        print(f"❌ COCO图像目录不存在: {args.coco_images_dir}")
        return
    
    if args.verify_only:
        # 仅验证数据
        verify_processed_data(args.output_dir, args.coco_images_dir)
    else:
        # 处理数据
        process_aokvqa_dataset(args.input_dir, args.output_dir, args.coco_images_dir)
        
        # 验证处理后的数据
        verify_processed_data(args.output_dir, args.coco_images_dir)

if __name__ == "__main__":
    main()

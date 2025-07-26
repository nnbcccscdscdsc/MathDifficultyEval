#!/usr/bin/env python3
"""
统一数学评估脚本
支持多种模型和配置的数学评估，包含智能模型检测和远程回退
"""

import argparse
import os
import sys
import openai
import logging
from math_evaluation_framework import MathEvaluationFramework, MODEL_CONFIGS

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 关闭OpenAI库的调试日志
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def check_local_model_availability(model_name: str) -> bool:
    """检查本地模型是否可用"""
    try:
        from transformers import AutoTokenizer
        # 尝试加载tokenizer来检查模型是否存在
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False  # 允许从缓存加载
        )
        logger.info(f"✅ 本地模型 {model_name} 可用")
        return True
        
    except Exception as e:
        logger.warning(f"❌ 本地模型 {model_name} 不可用: {e}")
        return False

def test_remote_model_connection(model_name: str, hf_token: str) -> bool:
    """测试远程模型连接"""
    try:
        if not hf_token:
            logger.error("❌ 未设置HF_TOKEN，无法测试远程模型")
            return False
        
        # 设置OpenAI客户端
        openai.api_key = hf_token
        openai.api_base = "https://router.huggingface.co/v1"
        
        # 为70B模型添加:novita后缀
        if "70B" in model_name:
            model_name_with_suffix = f"{model_name}:novita"
        else:
            model_name_with_suffix = model_name
        
        # 发送简单测试请求
        test_prompt = "What is 2 + 2?"
        completion = openai.ChatCompletion.create(
            model=model_name_with_suffix,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=50,
            temperature=0.1,
            top_p=0.9
        )
        
        logger.info(f"✅ 远程模型 {model_name_with_suffix} 连接正常")
        return True
        
    except Exception as e:
        logger.warning(f"❌ 远程模型 {model_name} 连接失败: {e}")
        return False

def select_best_model(model_key: str, hf_token: str = None) -> tuple:
    """选择最佳可用模型"""
    logger.info(f"🔍 检测模型可用性，首选: {model_key}")
    
    # 远程模型映射
    remote_models = {
        "deepseek_r1_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek_r1_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek_r1_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek_r1_70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    }
    
    # 70B模型直接走远端API，跳过本地检查
    if model_key == "deepseek_r1_70b" and hf_token:
        logger.info("🚀 70B模型直接使用远端API")
        remote_model_name = remote_models[model_key]
        if test_remote_model_connection(remote_model_name, hf_token):
            remote_config = {
                "name": remote_model_name,
                "type": "remote",
                "description": f"远程{model_key}模型"
            }
            return ("remote", remote_config)
        else:
            raise ValueError("❌ 70B远端模型连接失败")
    
    # 其他模型先检查本地，再检查远程
    # 检查本地模型
    if model_key in MODEL_CONFIGS:
        local_model_name = MODEL_CONFIGS[model_key]['name']
        if check_local_model_availability(local_model_name):
            return ("local", MODEL_CONFIGS[model_key])
    
    # 检查远程模型
    if model_key in remote_models and hf_token:
        remote_model_name = remote_models[model_key]
        if test_remote_model_connection(remote_model_name, hf_token):
            # 创建远程模型配置
            remote_config = {
                "name": remote_model_name,
                "type": "remote",
                "description": f"远程{model_key}模型"
            }
            return ("remote", remote_config)
    
    # 尝试其他模型大小
    model_sizes = ["7b", "14b", "32b", "70b"]
    for size in model_sizes:
        # 检查本地
        test_key = f"deepseek_r1_{size}"
        if test_key in MODEL_CONFIGS:
            local_model_name = MODEL_CONFIGS[test_key]['name']
            if check_local_model_availability(local_model_name):
                return ("local", MODEL_CONFIGS[test_key])
        
        # 检查远程
        if test_key in remote_models and hf_token:
            remote_model_name = remote_models[test_key]
            if test_remote_model_connection(remote_model_name, hf_token):
                remote_config = {
                    "name": remote_model_name,
                    "type": "remote",
                    "description": f"远程{test_key}模型"
                }
                return ("remote", remote_config)
    
    raise ValueError("❌ 没有可用的模型")

def main():
    parser = argparse.ArgumentParser(description="统一数学评估脚本")
    parser.add_argument("-m", "--model", type=str, default="deepseek_r1_7b", 
                       help="模型名称")
    parser.add_argument("-s", "--samples", type=int, default=50, 
                       help="样本数量")
    parser.add_argument("--no-openai", action="store_true", 
                       help="禁用OpenAI评估，只进行模型生成测试")
    parser.add_argument("--dataset", type=str, default="data/processed/fixed_200_samples.csv",
                       help="数据集路径")
    parser.add_argument("--hf-token", type=str, 
                       help="Hugging Face API Token")
    
    args = parser.parse_args()
    
    print("🚀 开始增强版统一数学评估")
    print(f"🤖 首选模型: {args.model}")
    print(f"📊 样本数量: {args.samples}")
    if args.no_openai:
        print("🚫 OpenAI评估: 已禁用")
    else:
        print("✅ OpenAI评估: 已启用")
    print("=" * 50)
    
    # 获取HF Token
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        print(f"🔑 HF Token: {hf_token[:10]}...{hf_token[-4:]}")
    else:
        print("⚠️ 未设置HF_TOKEN，将只使用本地模型")
    
    # 智能选择最佳模型
    try:
        model_type, model_config = select_best_model(args.model, hf_token)
        print(f"✅ 选择模型类型: {model_type}")
        print(f"📋 模型信息: {model_config['description']}")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # 检查数据集
    if not os.path.exists(args.dataset):
        print(f"❌ 数据集不存在: {args.dataset}")
        return
    
    print(f"✅ 找到样本文件: {args.dataset}")
    
    # 设置OpenAI API密钥
    openai_api_key = None
    if not args.no_openai:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("⚠️ 未设置OPENAI_API_KEY环境变量")
            print("请设置环境变量: export OPENAI_API_KEY='your-api-key'")
            print("或者使用 --no-openai 参数禁用OpenAI评估")
            return
    
    print("\n🔧 创建评估框架...")
    
    # 根据模型类型创建不同的评估框架
    if model_type == "local":
        # 使用本地模型
        framework = MathEvaluationFramework(
            model_config=model_config,
            openai_api_key=openai_api_key,
            max_samples=args.samples
        )
    else:
        # 使用远程模型
        print("🚀 使用远程模型评估...")
        
        # 创建远程评估框架
        class RemoteMathEvaluationFramework:
            def __init__(self, model_config, openai_api_key=None, max_samples=200):
                self.model_config = model_config
                self.model_name = model_config['name']
                self.max_samples = max_samples
                self.openai_api_key = openai_api_key
                
                # 设置日志
                import logging
                self.logger = logging.getLogger(__name__)
                
                # 生成运行标识符
                from datetime import datetime
                import random
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
                self.run_id = f"{timestamp}_{random_suffix}"
                self.logger.info(f"🆔 本次运行ID: {self.run_id}")
                
                # 设置远程API
                import openai
                self.openai_client = openai
                if openai_api_key:
                    openai.api_key = openai_api_key
                
                # 设置HF API
                self.hf_token = os.getenv("HF_TOKEN")
                if self.hf_token:
                    openai.api_key = self.hf_token
                    openai.api_base = "https://router.huggingface.co/v1"
            
            def load_dataset(self, dataset_path):
                """加载数据集"""
                import pandas as pd
                df = pd.read_csv(dataset_path)
                dataset = []
                for _, row in df.iterrows():
                    sample = {
                        'id': row['id'],
                        'problem': row['problem'],
                        'solution': row['solution'],
                        'answer': row['answer'],
                        'difficulty': row['difficulty'],
                        'topic': row['topic'],
                        'difficulty_score': row['difficulty_score']
                    }
                    dataset.append(sample)
                return dataset[:self.max_samples]  # 限制样本数量
            
            def generate_response(self, problem):
                """使用远程模型生成回答"""
                try:
                    # DeepSeek-R1推荐的提示格式
                    prompt = f"<think>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n</think>\n\n{problem}\n\n<think>\n"
                    
                    # 确保使用正确的API设置
                    original_api_base = self.openai_client.api_base
                    original_api_key = self.openai_client.api_key
                    
                    # 使用HF Router API (与32B成功配置一致)
                    self.openai_client.api_base = "https://router.huggingface.co/v1"
                    self.openai_client.api_key = self.hf_token
                    
                    # 为70B模型添加:novita后缀（与32B成功配置一致）
                    model_name_with_suffix = f"{self.model_name}:novita"
                    
                    completion = self.openai_client.ChatCompletion.create(
                        model=model_name_with_suffix,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.1,  # 降低温度，提高稳定性
                        top_p=0.9
                    )
                    
                    # 恢复原始设置
                    self.openai_client.api_base = original_api_base
                    self.openai_client.api_key = original_api_key
                    
                    response = completion.choices[0].message.content
                    return response.replace(prompt, "").strip()
                    
                except Exception as e:
                    self.logger.error(f"❌ 远程生成失败: {e}")
                    return f"生成失败: {str(e)}"
            
            def evaluate_with_openai(self, problem, model_response, correct_answer, standard_solution):
                """使用OpenAI评估模型回答，包含标准解法参考"""
                if not self.openai_api_key:
                    return {"error": "OpenAI客户端未初始化"}
                
                try:
                    # 临时切换到OpenAI API
                    original_api_base = self.openai_client.api_base
                    self.openai_client.api_base = "https://api.openai.com/v1"
                    self.openai_client.api_key = self.openai_api_key
                    
                    evaluation_prompt = f"""
You are a professional mathematical education evaluator. Please evaluate the quality of the answer to the following mathematical problem.

Problem: {problem}

Correct Answer: {correct_answer}

Standard Solution: {standard_solution}

Model Response: {model_response}

Please evaluate from the following aspects and give a score from 1 to 10:

1. Answer Correctness (1-10 points): Whether the final answer is correct
2. Reasoning Logic (1-10 points): Whether the reasoning process is clear and logical, compared to the standard solution
3. Step Completeness (1-10 points): Whether all necessary solution steps are shown, considering what the standard solution covers
4. Mathematical Accuracy (1-10 points): Whether mathematical calculations and formulas are accurate
5. Expression Clarity (1-10 points): Whether the expression is clear and easy to understand

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

Required JSON format:
{{
    "answer_correctness": <score>,
    "reasoning_logic": <score>,
    "step_completeness": <score>,
    "mathematical_accuracy": <score>,
    "expression_clarity": <score>,
    "overall_score": <average_of_all_scores>,
    "comments": "<brief_evaluation_comments>"
}}

Example:
{{
    "answer_correctness": 8,
    "reasoning_logic": 7,
    "step_completeness": 9,
    "mathematical_accuracy": 8,
    "expression_clarity": 7,
    "overall_score": 7.8,
    "comments": "Good reasoning but could be more detailed in steps"
}}
"""
                    
                    response = self.openai_client.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a professional mathematical education evaluator."},
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        temperature=0.3
                    )
                    
                    # 恢复HF API设置
                    self.openai_client.api_base = original_api_base
                    self.openai_client.api_key = self.hf_token
                    
                    response_content = response.choices[0].message.content
                    
                    # 解析JSON响应
                    import json
                    import re
                    
                    try:
                        evaluation = json.loads(response_content)
                        return evaluation
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"⚠️ JSON解析失败: {e}")
                        self.logger.warning(f"原始响应: {response_content[:200]}...")  # 只显示前200字符
                        
                        # 尝试提取JSON部分
                        try:
                            # 查找JSON对象
                            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group()
                                evaluation = json.loads(json_str)
                                self.logger.info(f"✅ 成功提取JSON部分")
                                return evaluation
                        except:
                            pass
                        
                        # 尝试修复常见的JSON格式问题
                        try:
                            # 移除可能的markdown代码块标记
                            cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                            evaluation = json.loads(cleaned_response)
                            self.logger.info(f"✅ 成功修复JSON格式")
                            return evaluation
                        except:
                            pass
                        
                        return {
                            "raw_response": response_content,
                            "error": "JSON parsing failed",
                            "parse_error": str(e),
                            "error_type": "json_decode_error"
                        }
                        
                except Exception as e:
                    self.logger.error(f"❌ OpenAI evaluation failed: {e}")
                    return {
                        "error": f"Evaluation failed: {e}",
                        "error_type": "openai_api_error",
                        "exception": str(e)
                    }
            
            def run_evaluation(self, dataset_path):
                """运行完整评估流程"""
                self.logger.info("🚀 开始完整评估流程")
                self.logger.info(f"📊 加载数据集: {dataset_path}")
                dataset = self.load_dataset(dataset_path)
                self.logger.info(f"✅ 加载了 {len(dataset)} 个样本")
                
                # 检查已有结果，实现断点续传
                start_index = self._check_existing_results()
                if start_index > 0:
                    self.logger.info(f"🔄 发现已有结果，从第 {start_index + 1} 个样本开始继续评估")
                    dataset = dataset[start_index:]
                else:
                    self.logger.info(f"🆕 开始全新评估")
                
                results = []
                successful_generations = 0
                successful_evaluations = 0
                
                self.logger.info(f"📝 开始评估 {len(dataset)} 个样本...")
                self.logger.info(f"⏱️ 预计需要时间: {len(dataset) * 30 / 60:.1f} 分钟（假设每个样本30秒）")
                
                from tqdm import tqdm
                for i, sample in enumerate(tqdm(dataset, desc="评估进度")):
                    global_index = start_index + i + 1
                    self.logger.info(f"\n--- 样本 {global_index}/351: {sample['id']} ---")
                    
                    # 生成模型回答
                    model_response = self.generate_response(sample['problem'])
                    
                    if model_response and not model_response.startswith("生成失败"):
                        successful_generations += 1
                        
                        # OpenAI评估
                        evaluation = self.evaluate_with_openai(
                            sample['problem'], 
                            model_response, 
                            sample['answer'],
                            sample['solution']  # 传入标准解法
                        )
                        
                        if isinstance(evaluation, dict) and "error" not in evaluation:
                            successful_evaluations += 1
                            self.logger.info(f"✅ 评估完成，总分: {evaluation.get('overall_score', 0):.2f}")
                        else:
                            # 详细错误信息处理
                            if isinstance(evaluation, dict):
                                error_msg = evaluation.get('error', '未知错误')
                                parse_error = evaluation.get('parse_error', '')
                                raw_response = evaluation.get('raw_response', '')[:200]  # 只显示前200字符
                                
                                if parse_error:
                                    self.logger.warning(f"⚠️ 评估失败: {error_msg}")
                                    self.logger.warning(f"解析错误: {parse_error}")
                                    if raw_response:
                                        self.logger.warning(f"原始响应: {raw_response}...")
                                else:
                                    self.logger.warning(f"⚠️ 评估失败: {error_msg}")
                            else:
                                self.logger.warning(f"⚠️ 评估失败: {str(evaluation)}")
                            
                            # 为评估失败的样本创建特殊评估
                            failed_evaluation = {
                                "answer_correctness": 0,
                                "reasoning_logic": 0,
                                "step_completeness": 0,
                                "mathematical_accuracy": 0,
                                "expression_clarity": 0,
                                "overall_score": 0,
                                "comments": f"评估失败: {str(evaluation)}",
                                "error": "evaluation_failed"
                            }
                            evaluation = failed_evaluation
                        
                        # 保存结果
                        result = {
                            "id": sample['id'],
                            "problem": sample['problem'],
                            "correct_answer": sample['answer'],
                            "standard_solution": sample['solution'],  # 添加原始解法
                            "model_response": model_response,
                            "difficulty": sample['difficulty'],
                            "topic": sample['topic'],
                            "evaluation": evaluation,
                            "generation_status": "success"
                        }
                        results.append(result)
                    else:
                        self.logger.warning(f"❌ 生成失败: {model_response}")
                        
                        # 为生成失败的样本创建特殊评估
                        failed_evaluation = {
                            "answer_correctness": 0,
                            "reasoning_logic": 0,
                            "step_completeness": 0,
                            "mathematical_accuracy": 0,
                            "expression_clarity": 0,
                            "overall_score": 0,
                            "comments": f"生成失败: {model_response}",
                            "error": "generation_failed"
                        }
                        
                        result = {
                            "id": sample['id'],
                            "problem": sample['problem'],
                            "correct_answer": sample['answer'],
                            "standard_solution": sample['solution'],  # 添加原始解法
                            "model_response": model_response,
                            "difficulty": sample['difficulty'],
                            "topic": sample['topic'],
                            "evaluation": failed_evaluation,
                            "generation_status": "failed"
                        }
                        results.append(result)
                    
                    # 每10个样本保存一次中间结果
                    if (i + 1) % 10 == 0:
                        global_count = start_index + i + 1
                        self.save_intermediate_results(results, global_count)
                        self.logger.info(f"💾 已处理 {global_count}/351 个样本，中间结果已保存")
                
                # 计算最终统计信息
                if results:
                    scores = [r['evaluation'].get('overall_score', 0) for r in results if r.get('generation_status') == 'success' and 'overall_score' in r['evaluation']]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    
                    final_results = {
                        "results": results,
                        "summary": {
                            "total_evaluated": len(results),
                            "successful_generations": successful_generations,
                            "failed_generations": len(results) - successful_generations,
                            "generation_success_rate": successful_generations / len(results) if results else 0,
                            "evaluation_success_rate": successful_evaluations / len(results) if results else 0,
                            "average_overall_score": avg_score
                        }
                    }
                    
                    self.logger.info(f"\n📋 评估摘要:")
                    self.logger.info(f"总评估样本: {len(results)}")
                    self.logger.info(f"生成成功率: {successful_generations / len(results) * 100:.1f}%")
                    self.logger.info(f"平均总分: {avg_score:.2f}")
                    
                    return final_results
                else:
                    return {"error": "没有有效结果"}
            
            def save_intermediate_results(self, results, count):
                """保存中间结果"""
                import json
                import os
                from datetime import datetime
                
                # 创建模型专用目录
                model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
                model_dir = f"data/intermediate/{model_safe_name}"
                os.makedirs(model_dir, exist_ok=True)
                
                # 使用运行ID创建子目录
                run_dir = f"{model_dir}/{self.run_id}"
                os.makedirs(run_dir, exist_ok=True)
                
                filename = f"{run_dir}/intermediate_results_{count}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"💾 中间结果已保存: {filename}")
    
            def _check_existing_results(self):
                """检查已有结果，返回应该开始的样本索引"""
                import os
                import json
                import glob
                
                # 创建模型专用目录
                model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
                model_dir = f"data/intermediate/{model_safe_name}"
                
                if not os.path.exists(model_dir):
                    return 0
                
                # 查找所有运行目录
                run_dirs = glob.glob(f"{model_dir}/*")
                if not run_dirs:
                    return 0
                
                # 找到最新的运行目录
                latest_run_dir = max(run_dirs, key=os.path.getctime)
                
                # 查找该运行目录下的所有中间结果文件
                result_files = glob.glob(f"{latest_run_dir}/intermediate_results_*.json")
                if not result_files:
                    return 0
                
                # 找到最大的样本数量
                max_count = 0
                for file_path in result_files:
                    try:
                        filename = os.path.basename(file_path)
                        # 提取文件名中的数字，如 intermediate_results_30.json -> 30
                        count_str = filename.replace('intermediate_results_', '').replace('.json', '')
                        count = int(count_str)
                        max_count = max(max_count, count)
                    except:
                        continue
                
                self.logger.info(f"📁 发现已有结果目录: {latest_run_dir}")
                self.logger.info(f"📊 已处理样本数量: {max_count}")
                
                return max_count
        
        # 创建远程评估框架实例
        framework = RemoteMathEvaluationFramework(
            model_config=model_config,
            openai_api_key=openai_api_key,
            max_samples=args.samples
        )
    
    # 运行评估
    try:
        results = framework.run_evaluation(args.dataset)
        print("🎉 评估完成！")
        return results
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 
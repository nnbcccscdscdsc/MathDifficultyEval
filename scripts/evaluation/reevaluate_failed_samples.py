#!/usr/bin/env python3
"""
重新评估失败样本脚本
专门用于重新评估因网络问题等导致的评估失败样本
"""

import json
import os
import glob
import logging
import openai
import time
from typing import List, Dict, Any
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FailedSampleReevaluator:
    def __init__(self, openai_api_key: str = None):
        """初始化重新评估器"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("❌ 未设置OPENAI_API_KEY环境变量")
        
        # 设置OpenAI客户端
        openai.api_key = self.openai_api_key
        self.openai_client = openai
        
        logger.info(f"🔑 OpenAI API Key: {self.openai_api_key[:10]}...{self.openai_api_key[-4:]}")
    
    def find_intermediate_files(self, model_name: str, run_id: str = None) -> List[str]:
        """查找指定模型的中间结果文件"""
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
        base_path = f"data/intermediate/{model_safe_name}"
        
        if run_id:
            # 指定运行ID
            run_path = f"{base_path}/{run_id}"
            if os.path.exists(run_path):
                pattern = os.path.join(run_path, "intermediate_results_*.json")
                files = glob.glob(pattern)
                logger.info(f"📁 找到指定运行ID的文件: {len(files)} 个")
                return sorted(files)
            else:
                logger.error(f"❌ 未找到运行ID: {run_id}")
                return []
        else:
            # 查找最新的运行ID
            if not os.path.exists(base_path):
                logger.error(f"❌ 未找到模型目录: {base_path}")
                return []
            
            # 获取所有运行ID目录
            run_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not run_dirs:
                logger.error(f"❌ 未找到任何运行目录")
                return []
            
            # 选择最新的运行ID
            latest_run = sorted(run_dirs)[-1]
            logger.info(f"📁 使用最新运行ID: {latest_run}")
            
            run_path = f"{base_path}/{latest_run}"
            pattern = os.path.join(run_path, "intermediate_results_*.json")
            files = glob.glob(pattern)
            logger.info(f"📁 找到文件: {len(files)} 个")
            return sorted(files)
    
    def load_all_results(self, files: List[str]) -> List[Dict[str, Any]]:
        """加载所有中间结果文件"""
        if not files:
            return []
        
        # 只加载最新的文件，避免重复加载
        latest_file = files[-1]
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ 加载最新文件: {os.path.basename(latest_file)} ({len(data)} 个样本)")
                return data
        except Exception as e:
            logger.error(f"❌ 加载文件失败 {latest_file}: {e}")
            return []
    
    def identify_failed_samples(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别评估失败的样本"""
        failed_samples = []
        
        for i, result in enumerate(results):
            evaluation = result.get('evaluation', {})
            
            # 检查是否有错误信息
            if isinstance(evaluation, dict):
                if 'error' in evaluation:
                    failed_samples.append({
                        'index': i,
                        'sample_id': result.get('id', f'unknown_{i}'),
                        'result': result,
                        'error': evaluation.get('error', 'unknown_error')
                    })
                    logger.info(f"❌ 发现失败样本 {i}: {result.get('id', 'unknown')} - {evaluation.get('error', 'unknown_error')}")
                elif evaluation.get('overall_score', 0) == 0 and 'comments' in evaluation:
                    # 检查是否是生成失败或评估失败
                    comments = evaluation.get('comments', '')
                    if '生成失败' in comments or '评估失败' in comments:
                        failed_samples.append({
                            'index': i,
                            'sample_id': result.get('id', f'unknown_{i}'),
                            'result': result,
                            'error': comments
                        })
                        logger.info(f"❌ 发现失败样本 {i}: {result.get('id', 'unknown')} - {comments}")
        
        logger.info(f"📊 总共发现 {len(failed_samples)} 个失败样本")
        return failed_samples
    
    def reevaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """重新评估单个样本"""
        try:
            # 使用OpenAI评估
            evaluation = self.evaluate_with_openai(
                sample['problem'],
                sample['model_response'],
                sample['correct_answer'],
                sample.get('standard_solution', '')
            )
            
            if isinstance(evaluation, dict) and 'error' not in evaluation:
                logger.info(f"✅ 重新评估成功: {sample.get('id', 'unknown')} - 总分: {evaluation.get('overall_score', 0):.2f}")
                return evaluation
            else:
                logger.warning(f"⚠️ 重新评估失败: {sample.get('id', 'unknown')} - {evaluation}")
                return evaluation
                
        except Exception as e:
            logger.error(f"❌ 重新评估异常: {sample.get('id', 'unknown')} - {e}")
            return {
                "error": f"Reevaluation failed: {e}",
                "error_type": "reevaluation_exception"
            }
    
    def evaluate_with_openai(self, problem: str, model_response: str, correct_answer: str, standard_solution: str = "") -> Dict[str, Any]:
        """使用OpenAI评估模型回答"""
        try:
            # 构建评估提示
            if standard_solution:
                evaluation_prompt = f"""
Please evaluate the quality of the answer to the following mathematical problem. You have access to the standard solution for reference.

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

CRITICAL: In the "comments" field, avoid using backslashes (\) or special characters that could break JSON parsing. Use simple text only.

Please return the evaluation result in JSON format:
{{
    "answer_correctness": score,
    "reasoning_logic": score,
    "step_completeness": score,
    "mathematical_accuracy": score,
    "expression_clarity": score,
    "overall_score": total_score/5,
    "comments": "Detailed evaluation with reference to standard solution"
}}
"""
            else:
                evaluation_prompt = f"""
Please evaluate the quality of the answer to the following mathematical problem.

Problem: {problem}

Correct Answer: {correct_answer}

Model Response: {model_response}

Please evaluate from the following aspects and give a score from 1 to 10:

1. Answer Correctness (1-10 points): Whether the final answer is correct
2. Reasoning Logic (1-10 points): Whether the reasoning process is clear and logical
3. Step Completeness (1-10 points): Whether all solution steps are shown
4. Mathematical Accuracy (1-10 points): Whether mathematical calculations and formulas are accurate
5. Expression Clarity (1-10 points): Whether the expression is clear and easy to understand

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

CRITICAL: In the "comments" field, avoid using backslashes (\) or special characters that could break JSON parsing. Use simple text only.

Please return the evaluation result in JSON format:
{{
    "answer_correctness": score,
    "reasoning_logic": score,
    "step_completeness": score,
    "mathematical_accuracy": score,
    "expression_clarity": score,
    "overall_score": total_score/5,
    "comments": "Detailed evaluation"
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
            
            response_content = response.choices[0].message.content
            
            # 解析JSON响应
            import re
            try:
                evaluation = json.loads(response_content)
                return evaluation
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON解析失败: {e}")
                
                # 尝试提取JSON部分
                try:
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        evaluation = json.loads(json_str)
                        logger.info(f"✅ 成功提取JSON部分")
                        return evaluation
                except:
                    pass
                
                # 尝试修复常见的JSON格式问题
                try:
                    cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                    evaluation = json.loads(cleaned_response)
                    logger.info(f"✅ 成功修复JSON格式")
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
            logger.error(f"❌ OpenAI evaluation failed: {e}")
            return {
                "error": f"Evaluation failed: {e}",
                "error_type": "openai_api_error",
                "exception": str(e)
            }
    
    def update_results(self, all_results: List[Dict[str, Any]], failed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """更新结果列表，替换失败的评估"""
        updated_results = all_results.copy()
        
        for failed_info in failed_samples:
            index = failed_info['index']
            original_result = failed_info['result']
            
            # 重新评估
            new_evaluation = self.reevaluate_sample(original_result)
            
            # 更新结果
            updated_results[index]['evaluation'] = new_evaluation
            
            # 添加重新评估标记
            updated_results[index]['reevaluated'] = True
            updated_results[index]['reevaluation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"🔄 已更新样本 {index}: {original_result.get('id', 'unknown')}")
        
        return updated_results
    
    def save_updated_results(self, updated_results: List[Dict[str, Any]], model_name: str, run_id: str = None) -> str:
        """保存更新后的结果"""
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
        base_path = f"data/intermediate/{model_safe_name}"
        
        if not run_id:
            # 使用最新的运行ID
            run_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            run_id = sorted(run_dirs)[-1]
        
        # 创建备份目录
        backup_dir = f"{base_path}/{run_id}_backup_{int(time.time())}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份原始文件
        original_files = glob.glob(f"{base_path}/{run_id}/intermediate_results_*.json")
        for file_path in original_files:
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 原始文件已备份到: {backup_dir}")
        
        # 重新组织结果到文件
        samples_per_file = 10
        file_count = 0
        
        for i in range(0, len(updated_results), samples_per_file):
            file_count += 1
            batch_results = updated_results[i:i + samples_per_file]
            
            # 保存到原文件位置
            file_path = f"{base_path}/{run_id}/intermediate_results_{i + len(batch_results)}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 保存更新文件: {os.path.basename(file_path)} ({len(batch_results)} 个样本)")
        
        return backup_dir
    
    def run_reevaluation(self, model_name: str, run_id: str = None):
        """运行完整的重新评估流程"""
        logger.info("🚀 开始重新评估失败样本")
        logger.info(f"🤖 模型: {model_name}")
        if run_id:
            logger.info(f"🆔 运行ID: {run_id}")
        
        try:
            # 1. 查找中间结果文件
            files = self.find_intermediate_files(model_name, run_id)
            if not files:
                logger.error("❌ 未找到中间结果文件")
                return
            
            # 2. 加载所有结果
            all_results = self.load_all_results(files)
            if not all_results:
                logger.error("❌ 未加载到任何结果")
                return
            
            # 3. 识别失败样本
            failed_samples = self.identify_failed_samples(all_results)
            if not failed_samples:
                logger.info("✅ 没有发现失败的样本")
                return
            
            # 4. 重新评估失败样本
            logger.info(f"🔄 开始重新评估 {len(failed_samples)} 个失败样本...")
            for failed_info in tqdm(failed_samples, desc="重新评估进度"):
                # 添加延迟避免API限制
                time.sleep(1)
            
            # 5. 更新结果
            updated_results = self.update_results(all_results, failed_samples)
            
            # 6. 保存更新后的结果
            backup_dir = self.save_updated_results(updated_results, model_name, run_id)
            
            logger.info("🎉 重新评估完成！")
            logger.info(f"📊 处理了 {len(failed_samples)} 个失败样本")
            logger.info(f"💾 原始文件备份到: {backup_dir}")
            
        except Exception as e:
            logger.error(f"❌ 重新评估失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="重新评估失败样本")
    parser.add_argument("--model", type=str, required=True, 
                       help="模型名称 (例如: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)")
    parser.add_argument("--run-id", type=str, 
                       help="运行ID (可选，默认使用最新的)")
    parser.add_argument("--openai-key", type=str, 
                       help="OpenAI API Key (可选，默认使用环境变量)")
    
    args = parser.parse_args()
    
    try:
        # 创建重新评估器
        reevaluator = FailedSampleReevaluator(args.openai_key)
        
        # 运行重新评估
        reevaluator.run_reevaluation(args.model, args.run_id)
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
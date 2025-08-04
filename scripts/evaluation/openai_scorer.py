#!/usr/bin/env python3
"""
OpenAI评分器：使用OpenAI API对数学答案进行专业评分
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
import openai
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging

class OpenAIScorer:
    """OpenAI评分器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化评分器"""
        self.config = ConfigLoader.load_config(config_path)
        self.openai_config = self.config.get('openai_scoring', {})
        
        # 设置OpenAI API
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("请设置环境变量 OPENAI_API_KEY")
        
        openai.api_key = api_key
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.model = self.openai_config.get('model', 'gpt-3.5-turbo')
        self.temperature = self.openai_config.get('temperature', 0.0)
        self.max_tokens = self.openai_config.get('max_tokens', 100)
        self.prompt_template = self.openai_config.get('prompt_template', '')
        
        self.logger.info(f"OpenAI评分器初始化完成，使用模型: {self.model}")
    
    def score_answer(self, problem: str, reference_answer: str, student_answer: str) -> Dict[str, Any]:
        """对单个答案进行评分"""
        try:
            # 构建提示词
            prompt = self.prompt_template.format(
                problem=problem,
                reference_answer=reference_answer,
                student_answer=student_answer
            )
            
            # 调用OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的数学教育评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 提取评分
            score_text = response.choices[0].message.content.strip()
            
            # 解析分数
            try:
                score = float(score_text)
                if score < 0 or score > 100:
                    self.logger.warning(f"分数超出范围: {score}, 设置为50")
                    score = 50.0
            except ValueError:
                self.logger.warning(f"无法解析分数: {score_text}, 设置为50")
                score = 50.0
            
            return {
                'openai_score': score,
                'score_text': score_text,
                'model_used': self.model,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI评分失败: {e}")
            return {
                'openai_score': 50.0,  # 默认分数
                'score_text': f"评分失败: {str(e)}",
                'model_used': self.model,
                'success': False,
                'error': str(e)
            }
    
    def score_batch(self, problems: List[Dict[str, Any]], delay: float = 1.0) -> List[Dict[str, Any]]:
        """批量评分"""
        self.logger.info(f"开始批量评分，共 {len(problems)} 个问题")
        
        results = []
        
        for i, problem_data in enumerate(problems):
            self.logger.info(f"评分进度: {i+1}/{len(problems)}")
            
            score_result = self.score_answer(
                problem=problem_data['problem'],
                reference_answer=problem_data['reference_answer'],
                student_answer=problem_data['student_answer']
            )
            
            # 合并结果
            result = {**problem_data, **score_result}
            results.append(result)
            
            # 添加延迟避免API限制
            if i < len(problems) - 1:
                time.sleep(delay)
        
        self.logger.info("批量评分完成")
        return results
    
    def analyze_scores_by_difficulty(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按难度等级分析分数"""
        difficulty_scores = {}
        
        for result in results:
            difficulty = result.get('difficulty', 'unknown')
            score = result.get('openai_score', 0)
            
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            
            difficulty_scores[difficulty].append(score)
        
        # 计算统计信息
        analysis = {}
        for difficulty, scores in difficulty_scores.items():
            if scores:
                analysis[difficulty] = {
                    'count': len(scores),
                    'mean_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'std_score': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5
                }
        
        return analysis
    
    def generate_score_report(self, results: List[Dict[str, Any]], model_name: str) -> str:
        """生成评分报告"""
        if not results:
            return "没有评分结果"
        
        # 按难度分析
        difficulty_analysis = self.analyze_scores_by_difficulty(results)
        
        # 计算总体统计
        all_scores = [r.get('openai_score', 0) for r in results]
        total_mean = sum(all_scores) / len(all_scores) if all_scores else 0
        
        report = f"""
# OpenAI评分报告 - {model_name}

## 总体统计
- 总样本数: {len(results)}
- 平均分数: {total_mean:.2f}
- 最高分数: {max(all_scores) if all_scores else 0:.2f}
- 最低分数: {min(all_scores) if all_scores else 0:.2f}

## 各难度等级统计
"""
        
        for difficulty, stats in difficulty_analysis.items():
            report += f"""
### {difficulty.upper()}
- 样本数: {stats['count']}
- 平均分数: {stats['mean_score']:.2f}
- 分数范围: {stats['min_score']:.2f} - {stats['max_score']:.2f}
- 标准差: {stats['std_score']:.2f}
"""
        
        return report

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI评分器测试")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    if args.test:
        # 测试评分器
        scorer = OpenAIScorer()
        
        test_cases = [
            {
                'problem': 'What is 2 + 3?',
                'reference_answer': '5',
                'student_answer': 'The answer is 5.',
                'difficulty': 'elementary'
            },
            {
                'problem': 'Solve for x: 2x + 5 = 13',
                'reference_answer': 'x = 4',
                'student_answer': '2x + 5 = 13\n2x = 8\nx = 4',
                'difficulty': 'middle'
            }
        ]
        
        results = scorer.score_batch(test_cases)
        
        for i, result in enumerate(results):
            print(f"测试 {i+1}:")
            print(f"  问题: {result['problem']}")
            print(f"  分数: {result['openai_score']}")
            print(f"  评语: {result['score_text']}")
            print()

if __name__ == "__main__":
    main() 
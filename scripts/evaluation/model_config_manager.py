#!/usr/bin/env python3
"""
模型配置管理器

用于加载和管理各个模型的独立配置文件
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class ModelConfigManager:
    """模型配置管理器"""
    
    def __init__(self, config_dir: str = "configs/models"):
        """初始化配置管理器"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 模型名称映射
        self.model_name_mapping = {
            "mistral-community/Mistral-7B-v0.2": "mistral-7b-v0.2.yaml",
            "lmsys/longchat-7b-16k": "longchat-7b-16k.yaml",
            "Yukang/LongAlpaca-13B-16k": "longalpaca-13b-16k.yaml",
            "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf": "oasst-llama-30b.yaml",
            "Yukang/LongAlpaca-70B-16k": "longalpaca-70b-16k.yaml"
        }
        
        # 缓存配置
        self._config_cache = {}
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        if model_name in self._config_cache:
            return self._config_cache[model_name]
        
        # 查找配置文件
        config_file = self.model_name_mapping.get(model_name)
        if not config_file:
            raise ValueError(f"未找到模型 {model_name} 的配置文件")
        
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self._config_cache[model_name] = config
            self.logger.info(f"加载模型配置: {model_name}")
            return config
            
        except Exception as e:
            self.logger.error(f"加载模型配置失败 {model_name}: {e}")
            raise
    
    def get_gpu_config(self, model_name: str) -> Dict[str, Any]:
        """获取GPU配置"""
        config = self.get_model_config(model_name)
        return config.get('gpu', {})
    
    def get_quantization_config(self, model_name: str, quantization: str = None) -> Dict[str, Any]:
        """获取量化配置"""
        config = self.get_model_config(model_name)
        quant_config = config.get('quantization', {})
        
        if quantization is None:
            quantization = quant_config.get('default', '4bit')
        
        return quant_config.get('options', {}).get(quantization, {})
    
    def get_generation_config(self, model_name: str) -> Dict[str, Any]:
        """获取生成配置"""
        config = self.get_model_config(model_name)
        return config.get('generation', {})
    
    def get_model_specific_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型特定配置"""
        config = self.get_model_config(model_name)
        return config.get('model_specific', {})
    
    def get_evaluation_config(self, model_name: str) -> Dict[str, Any]:
        """获取评估配置"""
        config = self.get_model_config(model_name)
        return config.get('evaluation', {})
    
    def get_prompt_template(self, model_name: str) -> str:
        """获取提示模板"""
        config = self.get_model_config(model_name)
        return config.get('prompt_template', "{problem}")
    
    def get_model_display_name(self, model_name: str) -> str:
        """获取模型显示名称"""
        config = self.get_model_config(model_name)
        return config.get('model', {}).get('display_name', model_name)
    
    def get_model_description(self, model_name: str) -> str:
        """获取模型描述"""
        config = self.get_model_config(model_name)
        return config.get('model', {}).get('description', "")
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """列出所有可用模型"""
        models = []
        for model_name, config_file in self.model_name_mapping.items():
            try:
                config = self.get_model_config(model_name)
                models.append({
                    'name': model_name,
                    'display_name': config.get('model', {}).get('display_name', model_name),
                    'description': config.get('model', {}).get('description', ""),
                    'gpu_count': config.get('gpu', {}).get('num_gpus', 1),
                    'config_file': config_file
                })
            except Exception as e:
                self.logger.warning(f"无法加载模型配置 {model_name}: {e}")
        
        return models
    
    def validate_model_config(self, model_name: str) -> bool:
        """验证模型配置"""
        try:
            config = self.get_model_config(model_name)
            
            # 检查必需字段
            required_fields = ['model', 'gpu', 'quantization', 'generation']
            for field in required_fields:
                if field not in config:
                    self.logger.error(f"模型配置缺少必需字段: {field}")
                    return False
            
            # 检查模型名称
            if 'name' not in config['model']:
                self.logger.error(f"模型配置缺少模型名称")
                return False
            
            # 检查GPU配置
            if 'num_gpus' not in config['gpu']:
                self.logger.error(f"GPU配置缺少GPU数量")
                return False
            
            self.logger.info(f"模型配置验证通过: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型配置验证失败 {model_name}: {e}")
            return False
    
    def print_model_info(self, model_name: str):
        """打印模型信息"""
        try:
            config = self.get_model_config(model_name)
            
            print(f"\n📋 模型信息: {model_name}")
            print("="*60)
            print(f"显示名称: {config['model']['display_name']}")
            print(f"描述: {config['model']['description']}")
            print(f"GPU数量: {config['gpu']['num_gpus']}")
            print(f"默认量化: {config['quantization']['default']}")
            print(f"最大输出长度: {config['generation']['max_new_tokens']}")
            print(f"评估样本数: {config['evaluation']['max_samples_per_run']}")
            print(f"超时时间: {config['evaluation']['timeout_seconds']}秒")
            
        except Exception as e:
            print(f"❌ 获取模型信息失败: {e}")
    
    def print_all_models(self):
        """打印所有模型信息"""
        print("\n🚀 可用模型列表")
        print("="*60)
        
        models = self.list_available_models()
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['display_name']}")
            print(f"   模型名称: {model['name']}")
            print(f"   描述: {model['description']}")
            print(f"   GPU数量: {model['gpu_count']}")
            print(f"   配置文件: {model['config_file']}")
            print()

def main():
    """主函数 - 用于测试配置管理器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型配置管理器")
    parser.add_argument("--list", action="store_true", help="列出所有模型")
    parser.add_argument("--info", type=str, help="显示指定模型的详细信息")
    parser.add_argument("--validate", type=str, help="验证指定模型的配置")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置管理器
    manager = ModelConfigManager()
    
    if args.list:
        manager.print_all_models()
    elif args.info:
        manager.print_model_info(args.info)
    elif args.validate:
        success = manager.validate_model_config(args.validate)
        if success:
            print(f"✅ 模型配置验证通过: {args.validate}")
        else:
            print(f"❌ 模型配置验证失败: {args.validate}")
    else:
        print("请使用 --list, --info 或 --validate 参数")

if __name__ == "__main__":
    main() 
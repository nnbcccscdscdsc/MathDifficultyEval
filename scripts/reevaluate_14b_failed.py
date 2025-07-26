#!/usr/bin/env python3
"""
重新评估14B模型失败样本的简化脚本
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.reevaluate_failed_samples import FailedSampleReevaluator

def main():
    """重新评估14B模型的失败样本"""
    
    # 14B模型名称
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    # 从日志中看到的运行ID
    run_id = "20250725_072638_vgyz"  # 根据你的实际运行ID修改
    
    print("🚀 开始重新评估14B模型失败样本")
    print(f"🤖 模型: {model_name}")
    print(f"🆔 运行ID: {run_id}")
    print("=" * 50)
    
    try:
        # 创建重新评估器
        reevaluator = FailedSampleReevaluator()
        
        # 运行重新评估
        reevaluator.run_reevaluation(model_name, run_id)
        
        print("\n🎉 重新评估完成！")
        print("💡 原始文件已自动备份")
        print("💡 失败样本已重新评估并更新")
        
    except Exception as e:
        print(f"❌ 重新评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
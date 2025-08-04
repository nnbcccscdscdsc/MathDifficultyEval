#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Difficulty Evaluation - Main Entry Point
数学难度评估系统主入口
"""

import os
import sys
from pathlib import Path

def print_menu():
    """打印主菜单"""
    print("=" * 60)
    print("🎯 Math Difficulty Evaluation System")
    print("数学难度评估系统")
    print("=" * 60)
    print("1. 📊 Generate Weighted Comparison Charts (生成权重对比图)")
    print("2. 📈 Generate Unified Model Analysis (生成统一模型分析)")
    print("3. 🔍 Run Math500 Evaluation (运行MATH-500评估)")
    print("4. 🔍 Run Hendrycks Math Evaluation (运行Hendrycks Math评估)")
    print("5. 🔍 Run DeepMath-103K Evaluation (运行DeepMath-103K评估)")
    print("6. 🛠️  Create Balanced Dataset (创建平衡数据集)")
    print("0. 🚪 Exit (退出)")
    print("=" * 60)

def run_script(script_path, description):
    """运行指定脚本"""
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"🚀 Running: {description}")
    print(f"📁 Script: {script_path}")
    print("-" * 40)
    
    try:
        # 添加scripts目录到Python路径
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        
        # 执行脚本
        exec(open(script_path).read())
        return True
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def main():
    """主函数"""
    while True:
        print_menu()
        
        try:
            choice = input("请选择功能 (Enter your choice): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                run_script("scripts/plotting/plot_all_datasets_weighted.py", 
                          "Generate Weighted Comparison Charts")
            elif choice == "2":
                run_script("scripts/plotting/unified_model_analysis.py", 
                          "Generate Unified Model Analysis")
            elif choice == "3":
                run_script("scripts/evaluation/unified_math_evaluation_math500.py", 
                          "Run Math500 Evaluation")
            elif choice == "4":
                run_script("scripts/evaluation/unified_math_evaluation_hendrycks.py", 
                          "Run Hendrycks Math Evaluation")
            elif choice == "5":
                run_script("scripts/evaluation/unified_math_evaluation.py", 
                          "Run DeepMath-103K Evaluation")
            elif choice == "6":
                run_script("scripts/utils/create_balanced_dataset.py", 
                          "Create Balanced Dataset")
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main() 
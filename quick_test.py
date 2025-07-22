#!/usr/bin/env python3
"""
快速测试脚本：验证项目基本功能
"""

import os
import sys
from pathlib import Path
import pandas as pd

def test_imports():
    """测试导入"""
    print("🔍 测试导入...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers导入失败: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas导入失败: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib导入失败: {e}")
        return False
    
    try:
        import seaborn
        print(f"✅ Seaborn: {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn导入失败: {e}")
        return False
    
    return True

def test_project_structure():
    """测试项目结构"""
    print("\n📁 测试项目结构...")
    
    required_files = [
        "scripts/data_processing.py",
        "scripts/model_evaluation.py", 
        "scripts/results_analysis.py",
        "scripts/utils.py",
        "configs/config.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "data",
        "data/raw",
        "data/processed", 
        "results",
        "models"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - 目录不存在")
            all_good = False
    
    return all_good

def test_config_loading():
    """测试配置加载"""
    print("\n⚙️ 测试配置加载...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from scripts.utils import ConfigLoader
        
        config = ConfigLoader.load_config()
        print("✅ 配置文件加载成功")
        
        # 检查必要的配置项
        required_sections = ['models', 'datasets', 'evaluation']
        for section in required_sections:
            if section in config:
                print(f"✅ 配置项 {section} 存在")
            else:
                print(f"❌ 配置项 {section} 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_data_processing():
    """测试数据处理"""
    print("\n📊 测试数据处理...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from scripts.data_processing import MathDataProcessor
        
        processor = MathDataProcessor()
        print("✅ 数据处理器初始化成功")
        
        # 创建测试数据
        test_data = [
            {
                'id': 'test_1',
                'problem': 'What is 2 + 3?',
                'solution': '5',
                'difficulty': 'elementary',
                'dataset': 'test'
            },
            {
                'id': 'test_2', 
                'problem': 'Solve for x: 2x + 5 = 13',
                'solution': 'x = 4',
                'difficulty': 'middle',
                'dataset': 'test'
            }
        ]
        
        df = pd.DataFrame(test_data)
        test_file = Path("data/processed/test_sample.csv")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(test_file, index=False)
        print("✅ 测试数据创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据处理测试失败: {e}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🖥️ 测试GPU可用性...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU可用: {gpu_name}")
            print(f"   GPU数量: {gpu_count}")
            print(f"   GPU内存: {gpu_memory:.2f} GB")
            return True
        else:
            print("⚠️  GPU不可用，将使用CPU")
            return True
            
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("🚀 数学难度评估项目 - 快速测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_imports),
        ("项目结构测试", test_project_structure),
        ("配置加载测试", test_config_loading),
        ("数据处理测试", test_data_processing),
        ("GPU可用性测试", test_gpu_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！项目可以正常运行。")
        print("\n📖 下一步操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行数据处理: python scripts/data_processing.py --sample")
        print("3. 运行模型评估: python scripts/model_evaluation.py --model llama-7b --dataset sample")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")
        print("\n🔧 建议:")
        print("1. 确保所有依赖已正确安装")
        print("2. 检查项目文件是否完整")
        print("3. 确保Python环境配置正确")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
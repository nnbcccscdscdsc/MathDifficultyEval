#!/usr/bin/env python3
"""
使用Hugging Face Inference Router测试DeepSeek-R1-Distill-Qwen-32B模型
回答2个数学问题
"""

import os
import openai
import time

def test_deepseek_32b():
    """测试DeepSeek 32B模型"""
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        print("❌ 未找到HF_TOKEN环境变量")
        print("请设置: export HF_TOKEN='your_huggingface_token'")
        return False
    
    print(f"🔑 HF Token: {api_key[:10]}...{api_key[-4:]}")
    print("🚀 开始测试DeepSeek-R1-Distill-Qwen-32B模型...")
    
    # 设置OpenAI客户端（使用旧版本API）
    openai.api_key = api_key
    openai.api_base = "https://router.huggingface.co/v1"
    
    # 测试问题列表
    test_questions = [
        "What is 15 * 7? Please put your answer in \\boxed{}.",
        "Solve for x: x^2 + 5x + 6 = 0. Please put your answer in \\boxed{}."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"📝 问题 {i}: {question}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            print("🔄 发送请求...")
            
            completion = openai.ChatCompletion.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:novita",
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                top_p=0.9
            )
            
            end_time = time.time()
            
            answer = completion.choices[0].message.content
            usage = completion.usage
            
            print(f"✅ 成功！耗时: {end_time - start_time:.2f}秒")
            print(f"📝 回答:\n{answer}")
            print(f"🔢 Token使用: {usage}")
                
        except Exception as e:
            print(f"❌ 请求失败: {e}")
    
    print(f"\n{'='*50}")
    print("🎉 测试完成！")
    print(f"{'='*50}")

if __name__ == "__main__":
    test_deepseek_32b() 
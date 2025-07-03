
import json
import sys
import os

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    # 生成简单的测试图
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.title('Python调用测试 - 正弦函数', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = sys.argv[1] if len(sys.argv) > 1 else 'test_output.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    # 返回结果
    result = {
        'success': True,
        'message': '基础Python调用测试成功',
        'image_path': output_path,
        'data_points': len(x),
        'file_exists': os.path.exists(output_path)
    }

    print(json.dumps(result, indent=2))

except Exception as e:
    result = {
        'success': False,
        'message': f'基础Python调用失败: {str(e)}',
        'error_type': type(e).__name__
    }
    print(json.dumps(result, indent=2))

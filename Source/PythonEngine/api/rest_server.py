from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
import numpy as np
from datetime import datetime
import threading
import queue
import time

# 导入具体的ML模块

from core.netcdf_handler import NetCDFHandler
from core.data_processor import DataProcessor
from visualization.field_generators import FieldGenerator
from utils.performance_utils import PerformanceMonitor

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局组件实例
lstm_predictor = LSTMCurrentPredictor()
pinn_model = PINNOceanModel()
netcdf_handler = NetCDFHandler()
data_processor = DataProcessor()
field_generator = FieldGenerator()
performance_monitor = PerformanceMonitor()

# 任务队列管理
task_queue = queue.Queue()
task_results = {}

def background_worker():
    """后台任务工作线程"""
    while True:
        try:
            task = task_queue.get()
            if task is None:
                break

            task_id = task['id']
            task_type = task['type']
            params = task['params']

            logger.info(f"开始执行任务 {task_id}: {task_type}")

            try:
                if task_type == 'lstm_training':
                    result = lstm_predictor.train_model(**params)
                elif task_type == 'pinn_training':
                    result = pinn_model.train_model(**params)
                elif task_type == 'current_prediction':
                    result = lstm_predictor.predict(**params)
                else:
                    result = {'error': f'未知任务类型: {task_type}'}

                task_results[task_id] = {
                    'status': 'completed',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                task_results[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

            task_queue.task_done()

        except Exception as e:
            logger.error(f"后台任务执行错误: {e}")

# 启动后台工作线程
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()

@app.route('/api/ml/predict_current_field', methods=['POST'])
def predict_current_field():
    """洋流场预测接口"""
    try:
        data = request.get_json()

        # 提取参数
        grid_data = np.array(data['grid_data'])
        time_range = data['time_range']
        coordinates = data['coordinates']
        model_type = data.get('model_type', 'LSTM')
        prediction_hours = data.get('prediction_hours', 24)

        # 执行预测
        start_time = time.time()
        prediction_result = lstm_predictor.predict(
            grid_data=grid_data,
            time_range=time_range,
            coordinates=coordinates,
            prediction_hours=prediction_hours
        )
        execution_time = time.time() - start_time

        # 格式化返回结果
        result = {
            'status': 'success',
            'prediction_data': prediction_result['velocity_field'].tolist(),
            'confidence_scores': prediction_result['confidence'].tolist(),
            'metadata': {
                'model_type': model_type,
                'prediction_hours': prediction_hours,
                'execution_time': execution_time,
                'grid_shape': grid_data.shape,
                'timestamp': datetime.now().isoformat()
            }
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"洋流场预测错误: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/ml/train_lstm', methods=['POST'])
def train_lstm():
    """LSTM模型训练接口"""
    try:
        data = request.get_json()

        # 生成任务ID
        task_id = f"lstm_train_{int(time.time())}"

        # 将训练任务加入队列
        task = {
            'id': task_id,
            'type': 'lstm_training',
            'params': {
                'dataset_path': data['dataset_path'],
                'model_config': data['model_config'],
                'validation_split': data.get('validation_split', 0.2),
                'callbacks': data.get('callbacks', [])
            }
        }

        task_queue.put(task)

        return jsonify({
            'status': 'accepted',
            'task_id': task_id,
            'message': '训练任务已提交，请使用task_id查询进度'
        })

    except Exception as e:
        logger.error(f"LSTM训练任务提交错误: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/ml/train_pinn', methods=['POST'])
def train_pinn():
    """PINN模型训练接口"""
    try:
        data = request.get_json()

        task_id = f"pinn_train_{int(time.time())}"

        task = {
            'id': task_id,
            'type': 'pinn_training',
            'params': {
                'physics_constraints': data['physics_constraints'],
                'domain_bounds': data['domain_bounds'],
                'boundary_conditions': data['boundary_conditions'],
                'network_architecture': data['network_architecture'],
                'loss_weights': data.get('loss_weights', {'data': 1.0, 'physics': 1.0}),
                'training_points': data['training_points']
            }
        }

        task_queue.put(task)

        return jsonify({
            'status': 'accepted',
            'task_id': task_id,
            'message': 'PINN训练任务已提交'
        })

    except Exception as e:
        logger.error(f"PINN训练任务提交错误: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/data/process_netcdf', methods=['POST'])
def process_netcdf():
    """NetCDF数据处理接口"""
    try:
        data = request.get_json()
        file_path = data['file_path']
        processing_options = data.get('processing_options', {})

        # 处理NetCDF文件
        result = netcdf_handler.process_file(file_path, **processing_options)

        # 数据质量控制
        if processing_options.get('quality_control', True):
            result = data_processor.quality_control(result)

        # 坐标转换
        if processing_options.get('coordinate_transform', True):
            result = data_processor.coordinate_transform(result)

        return jsonify({
            'status': 'success',
            'variable_count': len(result['variables']),
            'data_shape': result['data'].shape,
            'metadata': result['metadata'],
            'processing_time': result['processing_time']
        })

    except Exception as e:
        logger.error(f"NetCDF处理错误: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/visualization/generate', methods=['POST'])
def generate_visualization():
    """生成可视化数据接口"""
    try:
        data = request.get_json()
        simulation_result = data['simulation_result']
        viz_type = data.get('visualization_type', 'vector_field')

        # 生成可视化数据
        viz_data = field_generator.generate_field_visualization(
            simulation_result,
            visualization_type=viz_type,
            output_format=data.get('output_format', 'plotly_json'),
            resolution=data.get('resolution', {'width': 1920, 'height': 1080}),
            color_scheme=data.get('color_scheme', 'viridis')
        )

        return jsonify({
            'status': 'success',
            'visualization_data': viz_data,
            'metadata': {
                'type': viz_type,
                'resolution': data.get('resolution'),
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"可视化生成错误: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/tasks/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """查询任务状态接口"""
    if task_id in task_results:
        return jsonify(task_results[task_id])
    else:
        return jsonify({
            'status': 'pending',
            'message': '任务正在执行中或不存在'
        })

@app.route('/api/system/performance', methods=['GET'])
def get_performance():
    """获取系统性能指标接口"""
    try:
        metrics = performance_monitor.get_current_metrics()
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    logger.info("启动Python机器学习引擎REST API服务器")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

from flask import Flask, request, send_from_directory
from predict import predict
import pandas as pd
import json
import base64
import urllib.parse
import os
from flask_cors import CORS
import threading
import librosa
from predict_SED import bird_SED
from preprocess import generate_wave_spectrum
import datetime
import mysql.connector



app = Flask(__name__)
# 允许跨域访问
CORS(app)
lock = threading.Lock()  # 创建互斥锁对象

# 加载鸟鸣类别
label_dir = 'BirdsData-BirdsList.txt'
label_data = pd.read_csv(label_dir,sep="\t",header = None)
kinds = label_data.iloc[:,1].values
kinds = kinds.tolist()

# 格式化输出列表
def format_output(labels):
    result = []
    prev_label = None
    curr_label = None

    for i in range(len(labels)):
        prev_label = curr_label
        curr_label = labels[i]

        if curr_label != prev_label:
            result.append(curr_label)
        else:
            result.append(0)

    return result

# 格式化时间
def convert_seconds_to_hhmmss(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    time_str = str(time_obj)
    if time_obj.days > 0:
        time_str = f"{time_obj.days} days, {time_str}"
    return time_str

# 将预测结果转化为含有id，起始时间，终止时间，标签的dataframe，然后转化为csv文件
def generate_output_csv(labels, duration, filename):
    df = pd.DataFrame(columns=['id', 'start_time', 'end_time', 'label'])
    prev_label = None
    curr_label = None
    start_time = 0
    end_time = 0
    id = 0
    for i in range(len(labels)):
        if i != 0:
            prev_label = curr_label
        curr_label = labels[i]
        
        if curr_label != prev_label:
            if prev_label == None:
                continue
            if prev_label != 0 :
                end_time = i * duration
                df = df.append({'id': id, 'start_time': convert_seconds_to_hhmmss(start_time), 'end_time': convert_seconds_to_hhmmss(end_time), 'label': prev_label}, ignore_index=True)
                id += 1
            start_time = i * duration
        else:
            if curr_label == 0:
                end_time = i * duration
                df = df.append({'id': id, 'start_time': convert_seconds_to_hhmmss(start_time), 'end_time': convert_seconds_to_hhmmss(end_time), 'label': prev_label}, ignore_index=True)
                id += 1
    if labels[-1] != 0:
        end_time = len(labels) * duration
        df = df.append({'id': id, 'start_time': convert_seconds_to_hhmmss(start_time), 'end_time': convert_seconds_to_hhmmss(end_time), 'label': curr_label}, ignore_index=True)
    # 保存到csv文件
    df.to_csv('./csvfiles/'+filename + '.csv', index=False, encoding='utf-8-sig')

@app.route('/')
def index():
    return 'Hello World!'


# 鸟类检测与识别接口
@app.route('/process_data', methods=['GET'])
def process_data():
    # 获取文件名
    filename = request.args.get('filename')
    print(filename)
    
    # 切割音频片段--每个片段1秒
    audio, sr = librosa.load('upload/'+filename, sr=None)
    duration = len(audio) / sr
    segment_duration = 1  # 每个片段的时长（秒）
    segments = int(duration / segment_duration)
    
    # 结果列表
    result_list = []
    for i in range(segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration

        segment = audio[int(start_time * sr):int(end_time * sr)]  # 获取片段音频数据

        # 使用鸟鸣判断模型进行预测
        bird_detection_output = bird_SED(segment, sr) # 模型预测输出

        if bird_detection_output == 0:  # 判断为含有鸟鸣
            # 使用鸟鸣类型识别模型进行预测
            bird_type_output = predict(segment, sr)  # 模型预测输出
            # 将对应的鸟鸣类型添加到结果列表
            result_list.append(kinds[bird_type_output])
        else:
            # 将0（表示没有鸟鸣）添加到结果列表
            result_list.append(0)

    # 生成输出csv文件
    generate_output_csv(result_list, segment_duration, filename)
    # 格式化输出结果列表
    result_list = format_output(result_list)
    # 储存音频文件和标签到MySQL

    # 读取音频文件
    with open('upload/'+filename, 'rb') as f:
        audio_data = f.read()
        # 编码
        audio_data = base64.b64encode(audio_data).decode('utf-8')
    
    # MySQL参数
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="611611",
        database="bird_server"
    )
    # 连接数据库
    try:
        cursor = conn.cursor()
        # 插入鸟鸣文件和标签
        insert_sql = ("INSERT INTO bird_audio (audio_name, audio, audio_label) VALUES (%s, %s, %s)")
        cursor.execute(insert_sql, (filename, audio_data, str(result_list)))
        conn.commit()
        # 判断插入是否成功
        if cursor.rowcount > 0:
            print("插入成功")
            # 插入成功删除upload文件夹下的音频文件
            # os.remove('upload/'+filename)
        else:
            print("插入失败")
            
    except Exception as e:
        print("插入出错:", str(e))
        
    finally:
        # 关闭数据库连接
        cursor.close()
        conn.close()
    
    # 生成logmel_wave图片
    generate_wave_spectrum(filename, duration=3)
    
    # 返回结果列表
    
    # 返回logmel_wave.png图片
    with open('temp/wave_spectrum.png', 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)
        base64_data = base64_data.decode()
    # 返回结果列表csv文件
    with open('./csvfiles/'+filename+'.csv', 'rb') as f:
        csv_data = f.read()
        base64_csv = base64.b64encode(csv_data)
        base64_csv = base64_csv.decode('utf-8')
    
    json_data = json.dumps({'result': result_list, 'image': base64_data, 'csv': base64_csv})

    return json_data

# 获取图片列表接口
@app.route('/images', methods=['GET'])
def get_images():
    filename = request.args.get('filename')
    print(filename)
    page = int(request.args.get('page', 1))
    print('page',request.args.get('page'))
    # 每页一张图片
    per_page = int(request.args.get('per_page', 1))
    print('./images/'+filename)
    image_files = os.listdir('./images/'+filename)
    print('image_files',image_files)
    total_images = len(image_files)
    start_index = (page - 1) * per_page
    print('start_index',start_index)
    end_index = page * per_page
    print('end_index',end_index)
    images = image_files[start_index:end_index]
    image_data = []
    for image in images:
        print('image',image)
        image_url = f'images/{filename}/{image}'
        image_data.append(image_url)
    print('image_data',image_data)
    return json.dumps({'images': image_data, 'total_images': total_images})

# 下载图片接口
@app.route('/download_image', methods=['GET'])
def download_image():
    filename = request.args.get('filename')
    filename = urllib.parse.unquote(filename)
    dictionary = os.path.dirname(filename)
    filename = os.path.basename(filename)
    print(filename,dictionary)
    return send_from_directory(dictionary, filename)

# 下载csv文件接口
@app.route('/download_csv', methods=['GET'])
def download_csv():
    filename = request.args.get('filename')
    with open('./csvfiles/'+filename+'.csv', 'rb') as f:
        csv_data = f.read()
        base64_csv = base64.b64encode(csv_data)
        base64_csv = base64_csv.decode('utf-8')
    # json_data = json.dumps({'csv': base64_csv})
    return base64_csv

# 分片上传接口
@app.route('/upload', methods=['POST'])
def upload():
    # 从前端接收音频文件
    name = request.form.get('name')
    chunks_total = int(request.form.get('chunks_total'))
    filename = request.form.get('full_name')
    temp_path = 'upload/'+'chunks_'+filename
    # 使用互斥锁来确保只有一个请求可以创建文件夹
    with lock:
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

    fileStorage = request.files['file']  # 获取音频分片文件
    fileStorage.save(temp_path+'/' + name)
    return 'Chunk received!'

# 合并分片接口
@app.route('/merge', methods=['GET'])
def merge():
    # 获取文件名和分片总数
    filename = request.args.get('filename')
    chunks_total = int(request.args.get('chunks_total'))

    temp_path = 'upload/'+'chunks_'+filename
    # 检查是否已接收到所有分片文件
    if all(os.path.exists(os.path.join(temp_path+'/', f"{filename}_{i}")) for i in range(chunks_total)):
        print('All chunks received!')
        # 创建最终的 WAV 文件
        wav_path = os.path.join('upload/', filename)
        with open(wav_path, 'wb') as f:
            for i in range(chunks_total):
                with open(os.path.join(temp_path+'/', f"{filename}_{i}"), 'rb') as chunk:
                    f.write(chunk.read())
        # 合并后删除分片文件夹
        for i in range(chunks_total):
            os.remove(os.path.join(temp_path+'/', f"{filename}_{i}"))
        os.rmdir(temp_path)
        return 'Merge successfully!'
    else:
        return 'Merge failed!'


if __name__ == "__main__":
    app.run(host='192.168.1.200', port=5000, debug=True)

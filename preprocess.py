import os
from matplotlib import pyplot as plt
import pylab
import librosa
import librosa.display
import numpy as np



sr = 22050
duration = 2
def resample(sig, sr, duration):
    if sr != 22050:
        sig = librosa.resample(sig, sr, 22050)
    if len(sig) >= duration*sr:
        y1 = sig[:duration*sr]
    else:
        temp_y = np.zeros([duration*sr], np.float32)
        # print(temp_y.shape)
        for i, f in enumerate(sig):
            temp_y[i] = sig[i]
        y1 = temp_y
    return y1

# 重采样2，如果长度不够，就重复填充，截取前duration*sr
def resample2(sig, sr, duration):
    if sr != 44100:
        sig = librosa.resample(sig, sr, 44100)
    if len(sig) >= duration*sr:
        y1 = sig[:duration*sr]
    else:
        # 重复填充
        temp_y = np.tile(sig, duration*sr//len(sig)+1)
        y1 = temp_y[:duration*sr]
    return y1
        

def normalize(sig):
    sig = sig - np.mean(sig)
    sig = sig / np.max(np.abs(sig))
    return sig



def logmel(audio, fs):
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    sig = audio
    # fs, sig = read(file_path)
    # 幅度归一化
    sig = normalize(sig)
    # 重采样
    # sig = resample2(sig, sr, duration)
    # 计算mel谱
    S = librosa.feature.melspectrogram(y=sig, sr=fs, hop_length=512, win_length=2048, n_mels=13)
    # 转换为对数刻度
    S = librosa.power_to_db(S, ref=np.max)
    # # 求S的一阶二阶差分
    S1 = librosa.feature.delta(S)
    S2 = librosa.feature.delta(S, order=2)
    # # 拼接logmel特征
    S_concate = np.concatenate((S, S1, S2), axis=0)
    # # 可视化
    librosa.display.specshow(S_concate)
    # 以时间保存png
    plt.savefig('temp/logmel.png')
    plt.close()
    # 画出语音信号波形图
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    plt.plot(sig)
    plt.savefig('temp/wave.png')
    plt.close()
    # # 将wave和logmel画在一张图上，去掉白边
    # pylab.axis('off')  # no axis
    # pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    # plt.subplot(2,1,1)
    # plt.plot(sig)
    # pylab.axis('off')  # no axis
    # pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    # plt.subplot(2,1,2)
    # librosa.display.specshow(S)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.savefig('temp/logmel_wave.png')
    # 将wave和logmel画在一张图上，去掉白边
    fig = plt.gcf()
    fig.set_size_inches(18, 6)
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    plt.subplot(2,1,1)
    # 创建时间序列
    t = np.arange(len(sig)) / sr
    plt.plot(t,sig)
    plt.ylim([min(sig), max(sig)])
    plt.xlim([0, max(t)])  # 设置x轴范围从0到最大值
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    plt.subplot(2,1,2)
    S = librosa.amplitude_to_db(librosa.stft(sig))
    librosa.display.specshow(S, sr=sr, y_axis='log')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('temp/wave_spectrum.png',dpi=600)
    plt.close()


# 传入一段音频，每10s生成一张频谱图，并保持到./images/文件名_序号.png
def generate_wave_spectrum(filename, duration=3):
    audio_path = 'upload/'+filename
    audio, sr = librosa.load(audio_path, sr=None)
    # 不足n个duration*sr的补0
    if len(audio) % (duration*sr) != 0:
        temp_y = np.zeros([len(audio)+(duration*sr-len(audio) % (duration*sr))], np.float32)
        for i, f in enumerate(audio):
            temp_y[i] = audio[i]
        audio = temp_y
    
    image_path = 'images/'+filename
    isExists=os.path.exists(image_path)
    if not isExists:
        os.makedirs(image_path)
    # 每10s生成一张频谱图
    for i in range(len(audio)//(duration*sr)):
        sig = audio[i*duration*sr:(i+1)*duration*sr]
        fig = plt.gcf()
        fig.set_size_inches(18, 6)
        # pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
        plt.subplot(2,1,1)
        # 创建时间序列
        t = (np.arange(len(sig)) / sr) + i*duration
        plt.plot(t,sig)
        plt.ylim([min(sig), max(sig)])
        plt.xlim([min(t), max(t)])  # 设置x轴范围从0到最大值
        # plt.xlim([min(t), max(t)])  # 设置x轴范围从0到最大值
        # pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
        plt.subplot(2,1,2)
        S = librosa.amplitude_to_db(librosa.stft(sig))
        librosa.display.specshow(S, sr=sr, y_axis='log')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
        plt.savefig(image_path+'/'+filename+'_'+str(i)+'.png',dpi=600)
        plt.close()
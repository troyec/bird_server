import matplotlib.pyplot as plt
import numpy as np
import librosa
import pylab


def plot_audio_spectrum(audio_file):
    # 读取音频文件
    y, sr = librosa.load(audio_file)
    # 将wave和logmel画在一张图上，去掉白边
    fig = plt.gcf()
    fig.set_size_inches(18, 6)
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    plt.subplot(2,1,1)
    # 创建时间序列
    t = np.arange(len(y)) / sr
    plt.plot(t,y)
    plt.ylim([min(y), max(y)])
    plt.xlim([0, max(t)])  # 设置x轴范围从0到最大值
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
    plt.subplot(2,1,2)
    S = librosa.amplitude_to_db(librosa.stft(y))
    librosa.display.specshow(S, sr=sr, y_axis='log')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('temp/logmel_wave.png',dpi=600)
    return fig

audio_file = 'test.wav'
fig = plot_audio_spectrum(audio_file)
plt.show()  # 显示图形
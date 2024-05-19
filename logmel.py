import librosa
import librosa.display
import numpy as np


# input_path = r'D:\Choy\data\warblrb10k_ff1010bird'
# output_path = r'D:\Choy\data\features\logmel128_concat'
# file_dir = os.listdir(input_path)
sr = 22050
duration = 10
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
    if sr != 22050:
        sig = librosa.resample(sig, sr, 22050)
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

# for file in file_dir:
#     print(file)
#     pylab.axis('off')  # no axis
#     pylab.axes([0., 0., 1., 1.], frameon=True, xticks=[], yticks=[])  # Remove the white edge
#     sig, fs = librosa.load(input_path+'\\'+file)
#     sig = resample(sig, sr, duration)
#     S = librosa.feature.melspectrogram(y=sig, sr=fs, hop_length=512, win_length=2048, n_mels=128)
#     S = librosa.power_to_db(S, ref=np.max)
#     # 求S的一阶二阶差分
#     S1 = librosa.feature.delta(S)
#     S2 = librosa.feature.delta(S, order=2)
#     S = np.concatenate((S, S1, S2), axis=0)
#     save_path = output_path + '\\' + file + '.csv'

#     pd.DataFrame(S).to_csv(save_path, index=False, header=False)

#     pylab.close()

def logmel(audio, fs):
    sig = audio
    # 幅度归一化
    sig = normalize(sig)
    # 重采样
    sig = resample2(sig, sr, duration)
    # 计算mel谱
    # S = librosa.feature.melspectrogram(y=sig, sr=fs, hop_length=512, win_length=2048, n_mels=80)
    S = librosa.feature.melspectrogram(y=sig, sr=fs, hop_length=220, win_length=550, n_mels=13)
    # 转换为对数刻度
    S = librosa.power_to_db(S, ref=np.max)
    # librosa.display.specshow(S)
    # plt.show()
    # # 求S的一阶二阶差分
    # S1 = librosa.feature.delta(S)
    # S2 = librosa.feature.delta(S, order=2)
    # # 拼接logmel特征
    # S = np.concatenate((S, S1, S2), axis=0)

    # pd.DataFrame(S).to_csv(output_path, index=False, header=False)
    return S


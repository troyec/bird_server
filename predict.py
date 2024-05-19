import numpy as np
from preprocess import logmel
from extract_feature import get_deep_feature
import lightgbm as lgb




def predict(audio, sr):
    # 提取特征
    logmel(audio, sr)
    feat = get_deep_feature()
    # feat = pd.read_csv(temp_path,header = None).values
    values = np.float32(feat)
    test_data = values.flatten().reshape(1, -1)

    # 加载模型
    model = lgb.Booster(model_file='model/LightGBM_logmel_kind_20__1.txt')    
    # 预测
    predict_outcome = model.predict(test_data)
    print(predict_outcome)

    return np.argmax(predict_outcome)
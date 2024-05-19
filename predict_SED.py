import torch
from model_ghost_sknet import AudioClassifier
from logmel import logmel


def bird_SED(audio, sr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    model = AudioClassifier().to(device)
    model_weight_path = "./model/best_model_fold0.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    img = logmel(audio, sr)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).unsqueeze(0).float()
    
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # predict_labels.append(predict_cla)
        # predict_probs.append(predict[1])
        print(predict_cla)
    return predict_cla
    



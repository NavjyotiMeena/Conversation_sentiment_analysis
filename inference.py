from model import LSTMModel
import torch
import json

D_m = 250
D_g = 150
D_p = 150
D_e = 100
D_h = 100
D_a = 100

n_classes  = 7
dropout_val=0.5
model = LSTMModel(D_m, D_e, D_h,
                  n_classes=n_classes,
                  dropout=dropout_val)

model.load_state_dict(torch.load('./saved/model_lstm_new_data2.pth'))

 #####################################

from dataloader import DailyDialogueDataset2
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

 
def get_DailyDialogue_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    trainset = DailyDialogueDataset2('train', path)
    testset = DailyDialogueDataset2('test', path)
    validset = DailyDialogueDataset2('valid', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

 
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

 

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

 
#Loading data

preds=[]
labels_=[]
train_loader, valid_loader, test_loader = get_DailyDialogue_loaders('chatgpt_data2/dialogue.pkl')

# print(valid_loader)

for data in test_loader:
    # print(type(data))
    textf, qmask, umask, label = [d for d in data[:-1]]
    # print(textf)
    log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask) # seq_len, batch, n_classes
    lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
    labels_ = label.view(-1) # batch*seq_len
    pred_ = torch.argmax(lp_,1) # batch*seq_len
    print(pred_.data.cpu().numpy())
    print(len(pred_))

with open("data/train_9_test.json",'r') as f:
    json_object = json.load(f)


import pickle
emotion_decoder = pickle.load(open("chatgpt_data/emotion_label_decoder.pkl",'rb'))
print(pred_[0].data.cpu().numpy())
print(emotion_decoder[int(pred_[6].data.cpu())])


print("*************************")

i=0
for val,pred_val in zip(json_object['dialogue'],pred_):
    print(val)
    print('**********')
    print(pred_val)
    dictionary = {
        "Speaker": str(i%2),
        "Utternace": val['text'],
        "Predicted_Emotion": emotion_decoder[int(pred_val.data.cpu())]
    }
    i+=1
    # print(dictionary)
    json_object = json.dumps(dictionary, indent=4)
    with open("results_new_data_9.json", "a") as outfile:
        outfile.write(json_object)



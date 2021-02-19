import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
from vncorenlp import VnCoreNLP




class model_classify_IE(nn.Module):
    def __init__(self,num_cls):
        super().__init__()
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.clf = nn.Sequential(
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_cls)
        )
        self.activation = nn.Sigmoid()
    def forward(self,x):
        x = self.model(x)
        x = self.clf(x[1])
        return self.activation(x)

class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.rdrsegmenter = VnCoreNLP("weights/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    def __getitem__(self,idx):
        x = self.data[idx]
        x = self.tokenize(x) 
        sample ={
            "input":x
        }
        return sample
    def __len__(self):
        return len(self.data)
    def tokenize(self,text):
        try :
            sents = self.rdrsegmenter.tokenize(text)
            text_token = ' '.join([' '.join(sent) for sent in sents])
        except:
            print(text)
            text_token = ''
            print("fail")
        return text_token

class AlignCollate(object):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
    def __call__(self, batch):
        x = [sample['input'] for sample in batch]
        
        x = self.tokenizer(x,padding=True)["input_ids"]
        
        return torch.tensor(x)

class predict_phoBert(nn.Module):
    def __init__(self,path_model='weights/text_classification.pth'):
        super().__init__()
        self.model = model_classify_IE(3)
        self.model.load_state_dict(torch.load(path_model))
        self.collate_fn = AlignCollate()
    def forward(self,texts):
        test_gen = DataLoader(MyDataset(texts),batch_size =32,shuffle = False, num_workers = 0,collate_fn=self.collate_fn)
        x = next(iter(test_gen))
        out = self.model(x)
        out = torch.softmax(out,-1)
        return torch.argmax(out,-1)
                          

from transformers import AutoModel
from torch.utils import data
import sentence_transformers
from sentence_transformers import losses
from sklearn import linear_model
import numpy as np
import torch

class hybrid_model:
  def __init__(self,model_id):
    self.model = sentence_transformers.SentenceTransformer(model_id)
    self.classifier = linear_model.LogisticRegression(random_state=69)
  
  def train(self,text_list,label_list):
    train_examples = []
    for i in range(len(text_list)):
      train_examples.append(sentence_transformers.InputExample(texts=[text_list[i],''], label=[label_list[i]] ))

    train_dataloader = data.DataLoader(train_examples, shuffle=True, batch_size=10)

    train_loss = losses.SoftmaxLoss(self.model,sentence_embedding_dimension=768,num_labels=len(np.unique(label_list)))
    
    self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10,optimizer_class=torch.optim.AdamW) 
    
    train_x = self.model.encode(text_list)
    train_y = label_list
    
    self.classifier.fit(train_x,train_y)

    #print('worked')
  def predict(self,text_list):
    test_x = self.model.encode(text_list)
    
    pre = self.classifier.predict(test_x)

    return pre



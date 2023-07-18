import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np
from torch.utils.data import dataloader
import torch.nn as nn


model = GPT2LMHeadModel.from_pretrained("/data/jw/goorm_project3/model/except/1")

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')


def chat():
    
    with torch.no_grad():
        us=[]
        ch=[]
        cnt = 0 
        while 1:            
            q = input('나 > ').strip()
            us.append(q) # history 저장

            if q == 'quit':
                break        

            if cnt ==1:
                a= ' '
                user = "<usr>" + us[-2] + "<sys> " + ch[-1] + "<usr>" +  q  +  "<sys> " + "</s>"  + a
                # print("cnt = 1" ,us[-2], ch[-1])
                encoded = tokenizer.encode(user)
                input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
                output = model.generate(input_ids,max_length=50,
                                            num_beams=10, do_sample=False, 
                                            top_k=50, no_repeat_ngram_size=2,
                                            temperature=0.80)
                
                a = tokenizer.decode(output[0])
                temp_index = a.find('</s>')
                idx = torch.where(output[0]==tokenizer.encode('</s>')[0])
                idx = idx[0]+1
                output = output[0]
                chatbot = a[temp_index+4:-4]
                ch.append(chatbot)            
                cnt += 1
                print("챗봇 > {}".format(chatbot.strip()))
            
            elif cnt >= 2:
                a= ' '
                user = "<usr>" + us[-3] + "<sys> " + ch[-2] + "<usr>" + us[-2] + "<sys> " + ch[-1] + "<usr>" +  q  +  "<sys> " + "</s>"  + a
                # print("cnt = 2" ,us[-3], ch[-2])
                # print("cnt = 2" ,us[-2], ch[-1])
                encoded = tokenizer.encode(user)
                input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
                output = model.generate(input_ids, max_length=100,
                                            num_beams=10, do_sample=False, 
                                            top_k=50, no_repeat_ngram_size=2,
                                            temperature=0.70)
                
                a = tokenizer.decode(output[0])
                temp_index = a.find('</s>')
                idx = torch.where(output[0]==tokenizer.encode('</s>')[0])
                idx = idx[0]+1
                output = output[0]
                chatbot = a[temp_index+4:-4]
                ch.append(chatbot)            
                cnt += 1
                print("챗봇 > {}".format(chatbot.strip()))
            else:
                a= ' '
                user = "<usr>" + q + "<sys> " + "</s>"  + a 
                encoded = tokenizer.encode(user)
                input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
                output = model.generate(input_ids,max_length=50,
                                            num_beams=10, do_sample=False, 
                                            top_k=50, no_repeat_ngram_size=2,
                                            temperature=0.80)
                
                a = tokenizer.decode(output[0])
                temp_index = a.find('</s>')
                idx = torch.where(output[0]==tokenizer.encode('</s>')[0])
                idx = idx[0]+1
                output = output[0]
                chatbot = a[temp_index+4:-4]    
                ch.append(chatbot)        
                cnt += 1
                print("챗봇 > {}".format(chatbot.strip()))
                



chat()

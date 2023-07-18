import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np
from torch.utils.data import dataloader
from data.DataLoader import WellnessAutoRegressiveDataset
from data.DataLoader2 import WellnessAutoRegressiveDataset2
import torch.nn as nn

n_epoch = 5        # Num of Epoch
save_step = 100 # 학습 저장 주기
learning_rate = 5e-5  # Learning Rate
batch_size = 16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')


class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

  def generate(self,
               input_ids,
               do_sample=True,
               max_length= 60,
               top_p=0.92,
               top_k=50,
               temperature= 0.6,
               no_repeat_ngram_size = 2,
               num_return_sequences=2,
               early_stopping=True,
               ):
    
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_p =top_p,
               top_k=top_k,
               temperature=temperature,
               no_repeat_ngram_size= no_repeat_ngram_size,
               num_return_sequences=num_return_sequences,
               early_stopping = early_stopping,
              )

  def forward(self, input, labels = None):
    if labels is not None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)


    return outputs

model = DialogKoGPT2()
model.to(device)

loss_cross = torch.nn.CrossEntropyLoss(ignore_index=3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset= WellnessAutoRegressiveDataset2()
train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

losses = []
for epoch in range(n_epoch):
    count = 0
    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = torch.stack(data)  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
            data = data.transpose(1, 0)
            data= data.to(device)

            outputs = model(data, labels=data)
            _, logits = outputs[:2]
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = data[..., 1:].contiguous()

            loss = loss_cross(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())



            if count % save_step ==0 or (len(data) < batch_size):
                torch.save({
                    'epoch': epoch,
                    'train_no': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, f"/data/jw/goorm_project3/model/checkpoint/real_train_{epoch}")
                model.kogpt2.save_pretrained(f'/data/jw/goorm_project3/model/real_train/{epoch}')
            count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")
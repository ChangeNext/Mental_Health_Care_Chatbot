from transformers import PreTrainedTokenizerFast





tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

class WellnessAutoRegressiveDataset():
  """Wellness Auto Regressive Dataset"""

  def __init__(self,
               file_path1 = "/data/jw/goorm_project3/data/wellness_dialog_for_autoregressive.txt",
               file_path2 = "/data/jw/goorm_project3/data/ChatbotData.txt",
               n_ctx = 1024, token = tokenizer):
    self.file_path1 = file_path1
    self.file_path2 = file_path2
    self.data =[]
    self.q = []
    self.a = []
    self.tokenizer = token


    bos_token_id = [self.tokenizer.bos_token_id]
    eos_token_id = [self.tokenizer.eos_token_id]
    pad_token_id = [self.tokenizer.pad_token_id]

    file1 = open(self.file_path1, 'r', encoding = "utf-8")
    cnt = 0
    while True:
      line = file1.readline()
      if not line:
        break
      if cnt > 0:
        datas = line.split(".,")
        index_of_words = self.tokenizer.encode(datas[0]) + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
        self.q.append(self.tokenizer.encode(datas[0]))
        self.a.append(self.tokenizer.encode(datas[1][:-1]) + eos_token_id)
        pad_token_len = n_ctx - len(index_of_words)
        index_of_words += pad_token_id * pad_token_len
        self.data.append(index_of_words)
      cnt += 1
    file1.close()


    cnt1 = 0
    file2 = open(self.file_path2, 'r', encoding='utf-8')
    while True:
      line = file2.readline()
      if not line:
        break
      if cnt > 0:
        datas = line.split(",")
        index_of_words = self.tokenizer.encode(datas[0]) + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
        self.q.append(self.tokenizer.encode(datas[0]))
        self.a.append(self.tokenizer.encode(datas[1][:-1]) + eos_token_id)
        pad_token_len = n_ctx - len(index_of_words)
        index_of_words += pad_token_id * pad_token_len
        self.data.append(index_of_words)
      cnt1 += 1
    file2.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    # item = {}
    item = self.data[index]
    # label = self.a[index]
    # item["label"] = label
    
    return item

dataset = WellnessAutoRegressiveDataset()
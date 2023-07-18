from transformers import PreTrainedTokenizerFast

U_TKN = "<usr>"
S_TKN = "<sys>"
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

class WellnessAutoRegressiveDataset2():
  """Wellness Auto Regressive Dataset"""

  def __init__(self,
               file_path1 = "/data/jw/goorm_project3/data/wellness_dialog_for_autoregressive.txt",
               file_path2 = "/data/jw/goorm_project3/data/ChatbotData.txt",
               file_path3 = "/data/jw/goorm_project3/data/data.txt",
               file_path4 = '/data/jw/goorm_project3/data/data2.txt',
               file_path5 = '/data/jw/goorm_project3/data/data3.txt',
               n_ctx = 1024, token = tokenizer):
    self.file_path1 = file_path1
    self.file_path2 = file_path2
    self.file_path3 = file_path3
    self.file_path4 = file_path4
    self.file_path5 = file_path5
    self.data =[]
    self.q = []
    self.a = []
    self.tokenizer = token
    self.q_token = U_TKN
    self.a_token = S_TKN

    bos_token_id = [self.tokenizer.bos_token_id]
    eos_token_id = [self.tokenizer.eos_token_id]
    pad_token_id = [self.tokenizer.pad_token_id]
    ######################1############################
    file1 = open(self.file_path1, 'r', encoding = "utf-8")
    cnt = 0
    while True:
      line = file1.readline()
      if not line:
        break
      if cnt > 0:
        datas = line.split(".,")
        index_of_words = self.tokenizer.encode(self.q_token + datas[0]) + self.tokenizer.encode(self.a_token) + bos_token_id + self.tokenizer.encode(datas[1][:-1]) + eos_token_id
        pad_token_len = n_ctx - len(index_of_words)
        index_of_words += pad_token_id * pad_token_len
        self.data.append(index_of_words)
      cnt += 1
    file1.close()
    #######################2############################
    cnt1 = 0
    file2 = open(self.file_path2, 'r', encoding='utf-8')
    while True:
      line = file2.readline()
      if not line:
        break
      if cnt1 > 0:
        datas = line.split(",")
        index_of_words = self.tokenizer.encode(self.q_token + datas[0]) + self.tokenizer.encode(self.a_token) + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
        pad_token_len = n_ctx - len(index_of_words)
        index_of_words += pad_token_id * pad_token_len
        self.data.append(index_of_words)
      cnt1 += 1
    file2.close()

    #######################3############################
    file3 = open(self.file_path3, 'r', encoding='utf-8')
    while True:
      line = file3.readline()
      if not line:
        break
      datas = line.split(",")
      index_of_words = self.tokenizer.encode(self.q_token + datas[0]) + self.tokenizer.encode(self.a_token) + bos_token_id + self.tokenizer.encode(datas[1][:-1])+ eos_token_id
      pad_token_len = n_ctx - len(index_of_words)
      index_of_words += pad_token_id * pad_token_len
      self.data.append(index_of_words)
    file3.close()

  #######################4############################
    file4 = open(self.file_path4, 'r', encoding='utf-8')
    while True:
      line = file4.readline()
      if not line:
        break
      datas = line.split(",")
      index_of_words = self.tokenizer.encode(self.q_token + datas[0]) \
                  + self.tokenizer.encode(self.a_token+ datas[1]) \
                  + self.tokenizer.encode(self.q_token + datas[2]) \
                  + self.tokenizer.encode(self.a_token) + bos_token_id + self.tokenizer.encode(datas[3][:-1])+ eos_token_id
      pad_token_len = n_ctx - len(index_of_words)
      index_of_words += pad_token_id * pad_token_len
      self.data.append(index_of_words)
    file4.close()


    #######################5############################
    file5 = open(self.file_path5, 'r', encoding='utf-8')
    while True:
      line = file5.readline()
      if not line:
        break
      datas = line.split(",")
      index_of_words = self.tokenizer.encode(self.q_token + datas[0]) \
                  + self.tokenizer.encode(self.a_token+ datas[1]) \
                  + self.tokenizer.encode(self.q_token + datas[2]) \
                  + self.tokenizer.encode(self.a_token+ datas[3]) \
                  + self.tokenizer.encode(self.q_token + datas[4]) \
                  + self.tokenizer.encode(self.a_token) + bos_token_id + self.tokenizer.encode(datas[5][:-1])+ eos_token_id
      pad_token_len = n_ctx - len(index_of_words)
      index_of_words += pad_token_id * pad_token_len
      self.data.append(index_of_words)
    file5.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data[index]
    return item

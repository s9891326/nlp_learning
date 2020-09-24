"""
實作一個可以用來讀取訓練 / 測試集的 Dataset，這是你需要徹底了解的部分。
此 Dataset 每次將 tsv 裡的一筆成對句子轉換成 BERT 相容的格式，並回傳 3 個 tensors：
- tokens_tensor：兩個句子合併後的索引序列，包含 [CLS] 與 [SEP]
- segments_tensor：可以用來識別兩個句子界限的 binary tensor
- label_tensor：將分類標籤轉換成類別索引的 tensor, 如果是測試集則回傳 None
"""
import pandas as pd
import numpy as np
import torch
import pysnooper
from IPython.core.display import clear_output
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from transformers import BertForSequenceClassification


class FakeNewsDataset(Dataset):
    # 讀取前處理後的tsv檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {"agreed": 0, "disagreed": 1, "unrelated": 2}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    # @pysnooper.snoop()
    def __getitem__(self, idx):
        if self.mode == "test":
            text_1, text_2 = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_1, text_2, label = self.df.iloc[idx, :].values
            # 將 label 文字也轉換成索引方便轉換成 tensor
            # label_id = self.label_map[label]
            # label_tensor = torch.tensor(label_id)
            label_tensor = self.label_map[label]

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_1)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)

        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_2)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len

    def convert_ids2tokens(self, tokens_list: list):
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_list)
        return "".join(tokens)

"""
實作可以一次回傳一個 mini-batch 的 DataLoader
這個 DataLoader 吃我們上面定義的 `FakeNewsDataset`，
回傳訓練 BERT 時會需要的 4 個 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 測試集有labels
    if samples[0][2] is not None:
        label_ids = [s[2] for s in samples]
    else:
        label_ids = None

    # zero padding 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def print_data(data: list) -> None:
    for s in data:
        print(s[0] + " : " + str(s[1]))
    print("--------------------")

# 載入一個可以做中文多分類任務的模型，n_class = 3
def init_text_multi_classification_model():
    pretrained_model_name = "bert-base-chinese"
    num_labels = 3
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=num_labels)
    clear_output()

    # high-level 顯示此模型裡的 modules
    print("""
    name            module
    ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))

    return model

# 初始化一個每次回傳 64 個訓練樣本的 DataLoader
# 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
def init_train_data_loader(train_set):
    BATCH_SIZE = 64
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                             collate_fn=create_mini_batch)
    data = next(iter(train_loader))
    tokens, segments, masks, label = data
    tokens = np.array(tokens)
    segments = np.array(segments)
    masks = np.array(masks)
    label = np.array(label)
    print(f"""
        tokens.shape   = {tokens.shape}
        {tokens}
        ------------------------
        segments.shape = {segments.shape}
        {segments}
        ------------------------
        masks.shape    = {masks.shape}
        {masks}
        ------------------------
        label.shape        = {label.shape}
        {label}
        """)
    return train_loader

# 初始化bert模型，並跟原始文本比較
def init_bert_model_and_show_diff():
    # 記得我們是使用中文 BERT
    model_version = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_version)
    train_set = FakeNewsDataset("train", tokenizer=tokenizer)
    print("train_set = ", train_set)

    # 選擇第一個樣本
    sample_idx = 0

    # 將原始文本拿出做比較
    text_a, text_b, label = train_set.df.iloc[sample_idx].values

    # 利用剛剛建立的 Dataset 取出轉換後的 id tensors
    tokens_tensor, segments_tensor, label_tensor = train_set[sample_idx]

    # 將 tokens_tensor 還原成文本
    combined_text = train_set.convert_ids2tokens(tokens_tensor.tolist())

    # 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
    print("[原始文本]")
    print_data([["句子 1", text_a], ["句子 2", text_b], ["分類", label]])
    print("[Dataset 回傳的 tensors]")
    print_data([["tokens_tensor", tokens_tensor], ["segments_tensor", segments_tensor], ["label_tensor", label_tensor]])
    print("[還原 tokens_tensors]", combined_text)

    return train_set

"""
定義一個可以針對特定 DataLoader 取得模型預測結果以及分類準確度的函式
之後也可以用來生成上傳到 Kaggle 競賽的預測結果

2019/11/22 更新：在將 `tokens`、`segments_tensors` 等 tensors
丟入模型時，強力建議指定每個 tensor 對應的參數名稱，以避免 HuggingFace
更新 repo 程式碼並改變參數順序時影響到我們的結果。
"""
def get_predictions(model, data_loader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        # 遍巡整個資料集
        for data in data_loader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                # total += labels.size(0)
                total += len(labels)
                correct += (torch.tensor(labels) == pred).sum().item()

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

        if compute_acc:
            acc = correct / total
            return predictions, acc
        return predictions

def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]

def calculator_module_params(model):
    model_params = get_learnable_params(model)
    clf_params = get_learnable_params(model.classifier)

    print(f"""
    整個分類模型的參數量：{sum(p.numel() for p in model_params)}
    線性分類器的參數量：{sum(p.numel() for p in clf_params)}
    """)

def train_fine_tuning():
    # 訓練模型
    model.train()

    # 使用Adam optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 6

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data in train_loader:
            # tokens_tensors, segments_tensors, masks_tensors, labels = t.to(device) for t in data
            tokens_tensors, segments_tensors, masks_tensors, labels = data
            labels = torch.tensor(labels)

            # 將參數梯度歸零
            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)

            loss = outputs[0]

            # backward
            loss.backward()
            optimizer.step()

            # 紀錄當前batch loss
            running_loss += loss.item()

        # 計算分類準確率
        _, acc = get_predictions(model, train_loader, compute_acc=True)
        print("[epoch %d] loss: %.3f, acc: %.3f" % (epoch + 1, running_loss, acc))

def final_train(model):
    model_version = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_version)
    # 建立測試集。這邊我們可以用跟訓練時不同的 batch_size，看你 GPU 多大
    test_set = FakeNewsDataset("test", tokenizer=tokenizer)
    test_loader = DataLoader(test_set, batch_size=256, collate_fn=create_mini_batch)

    # 用分類模型預測測試集
    predictions = get_predictions(model, test_loader)

    # 用來將預測的 label id 轉回 label 文字
    index_map = {v: k for k, v in test_set.label_map.items()}

    # 生成 Kaggle 繳交檔案
    df = pd.DataFrame({"Category": predictions.tolist()})
    df['Category'] = df.Category.apply(lambda x: index_map[x])
    df_pred = pd.concat([test_set.df.loc[:, ["Id"]], df.loc[:, 'Category']], axis=1)
    df_pred.to_csv('bert_1_prec_training_samples.csv', index=False)
    df_pred.head()


if __name__ == "__main__":
    train_set = init_bert_model_and_show_diff()
    train_loader = init_train_data_loader(train_set)
    # 載入一個可以做中文多分類任務的模型，n_class = 3
    model = init_text_multi_classification_model()

    # 讓模型跑在 GPU 上並取得訓練集的分類準確率
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    model = model.to(device)
    # _, acc = get_predictions(model, train_loader, compute_acc=True)
    # print("classification acc = ", acc)

    calculator_module_params(model)

    # 訓練該下游任務模型
    # train_fine_tuning()

    # 對新樣本做推論
    final_train(model)

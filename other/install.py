import sys
# !test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"
# # !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
# !test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
# if not 'bertviz_repo' in sys.path:
#   sys.path += ['bertviz_repo']
# !pip install regex

from bertviz_repo.bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
from bertviz_repo.bertviz.head_view_bert import show
from bertviz_repo.bertviz import head_view

def call_html():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

bert_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(bert_version)
tokenizer = BertTokenizer.from_pretrained(bert_version, do_lower_case=do_lower_case)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
call_html()
show(model, tokenizer, sentence_a, sentence_b)

# 記得我們是使用中文 BERT
model_version = 'bert-base-chinese'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)

# 情境 1 的句子
sentence_a = "胖虎叫大雄去買漫畫，"
sentence_b = "回來慢了就打他。"

# 得到 tokens 後丟入 BERT 取得 attention
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
call_html()

# 交給 BertViz 視覺化
head_view(attention, tokens)

# 注意：執行這段程式碼以後只會顯示下圖左側的結果。
# 為了方便你比較，我把情境 2 的結果也同時附上
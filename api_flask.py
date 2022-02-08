import copy
import numpy as np
import string
from flask import Flask,jsonify,request
from util.search_synonyms import pog_tagging,get_synonyms
from util.load_mymodel import load_model
from util.deal_input import deal_sentence
from util.check_is_changed import convert,get_similarity_score
# from util.augmente_data import text_generation
# from util.gramformer import Gramformer
from sklearn.feature_extraction.text import CountVectorizer
# from bert4keras.backend import keras
# from bert4keras.models import build_transformer_model
# from bert4keras.tokenizers import Tokenizer
# from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
#
# maxlen = 510
vectorizer = CountVectorizer()
# summarizer = pipeline('summarization')
#gf = Gramformer(models=1,use_gpu=False) #1=corrector,2=detector

# # 模型配置
# config_path = r'./root/ckpt/chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = r'./root/ckpt/chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = r'./root/ckpt/chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt'
#
# # 建立分词器
# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
#
# # 建立加载模型
# roformer = build_transformer_model(
#     config_path,
#     checkpoint_path,
#     model='roformer',
#     application='unilm',
#     with_pool='linear'
# )
#
# encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
# seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])
#
#
# class SynonymsGenerator(AutoRegressiveDecoder):
#     """seq2seq解码器
#     """
#     @AutoRegressiveDecoder.wraps(default_rtype='probas')
#     def predict(self, inputs, output_ids, step):
#         token_ids, segment_ids = inputs
#         token_ids = np.concatenate([token_ids, output_ids], 1)
#         segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
#         return self.last_token(seq2seq).predict([token_ids, segment_ids])
#
#     def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
#         token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
#         for i in mask_idxs:
#             token_ids[i] = tokenizer._token_mask_id
#         output_ids = self.random_sample([token_ids, segment_ids], n,
#                                         topp=topp)  # 基于随机采样
#         return [tokenizer.decode(ids) for ids in output_ids]
#
#
# synonyms_generator = SynonymsGenerator(
#     start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
# )
#
#
# def gen_synonyms(text, n=100, k=10, mask_idxs=[]):
#     ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
#     做法：用seq2seq生成，并用encoder算相似度并排序。
#     '''
#     r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
#     r = [i for i in set(r) if i != text]
#     r = [text] + r
#     X, S = [], []
#     for t in r:
#         x, s = tokenizer.encode(t)
#         X.append(x)
#         S.append(s)
#     X = sequence_padding(X)
#     S = sequence_padding(S)
#     Z = encoder.predict([X, S])
#     Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
#     argsort = np.dot(Z[1:], -Z[0]).argsort()
#     return [r[i + 1] for i in argsort[:k]]



app = Flask(__name__)  # 创建一个服务，赋值给APP

# 加载模型
for i, j in zip(['en' for i in range(6)], ['zh', 'de', 'fr', 'it', 'es', 'sn']):
    path1 = f'./model_pickle/translation_{i}_{j}.pickle'
    path2 = f'./model_pickle/translation_{j}_{i}.pickle'
    globals()[f"translation_{i}_{j}"] = load_model(f"translation_{i}_{j}", path1)
    globals()[f"translation_{j}_{i}"] = load_model(f"translation_{j}_{i}", path2)

#回译
def translation(*args):
    """封装回译的函数"""
    translation = eval(f"translation_{args[0]}_{args[1]}")
    sentence = args[2]
    translated_text = translation(sentence, max_length=512)[0]['translation_text']

    translation1 = eval(f"translation_{args[1]}_{args[0]}")
    sentence_m = copy.deepcopy(translated_text)
    translated_text1 = translation1(sentence_m, max_length=512)[0]['translation_text']

    return translated_text1

#回译的接口                      没得问题
@app.route('/back_translation',methods=['post'])#指定接口访问的路径，支持什么请求方式get，post
def get_back_translation():
    sentence = request.form.get('sentence')
    parameters = request.form.get('parameters').split(',')
    if sentence==None:
        return '请输入英语句子'
    elif parameters == None:
        return '请输入参数'
    print(parameters)
    print(sentence)
    no_data = [('zh', 'fr'), ('zh', 'es'), ('fr', 'it'), ('es', 'zh'), ('it', 'zh'), ('fr', 'zh'), ('de', 'zh')]
    if len(parameters) == 1:
        use_model = [('en',parameters[0])]
    else:
        use_model = [('en',parameters[0])]
        n = len(parameters)
        for i in range(n):
            for j in range(i + 1, n):
                item = (parameters[i], parameters[j])
                if item in no_data:
                    return '重新输入你选择的语言'
                use_model.append(item)
        use_model.append((parameters[-1],'en'))
    print(f'本次用到的模型有{use_model}')
    # 回译
    # labels = {'zh': "中文", 'de': '德语', 'fr': '法语', 'it': '意大利语', 'es': '西班牙语'}
    n = 0
    for i in use_model:
        if n == 0:
            translated_text1 = translation(i[0],i[1], deal_sentence(sentence))
            continue
        translated_text1 = translation(i[0],i[1], deal_sentence(translated_text1))
    return jsonify({f"{','.join(parameters)}":translated_text1})

# #simbert生成同义句的接口                 没得问题
# @app.route('/simbert',methods=['post'])
# def get_simbert():
#     sentence = request.form.get('sentence')
#     if sentence==None:
#         return '请输入英语句子'
#     print(sentence)
#     # sim_bert
#     zh_sentence = eval('translation_en_zh')(sentence, max_length=512)[0]['translation_text']
#     print(zh_sentence)
#     simbert_sentence = [eval('translation_zh_en')(i, max_length=512)[0]['translation_text'] for i in gen_synonyms(zh_sentence)]
#     return jsonify(simbert_sentence)

#结构是否改变的接口                    没得问题
@app.route('/is_changed',methods=['post'])
def get_is_changed():
    sentence_source = request.form.get('sentence_source')
    sentence_changed = request.form.get('sentence_changed')
    if sentence_source==None or sentence_changed==None:
        return '请输入句子'
    result = convert(sentence_source,sentence_changed)
    return result

#获得句子的文本相似度得分的接口      没得问题
@app.route('/similarity_score',methods=['post'])
def similarity_score():
    sentence_source = request.form.get('sentence_source')
    sentence_changed = request.form.get('sentence_changed')
    print('原句子')
    print(sentence_source)
    print('生成的句子')
    print(sentence_changed)
    if sentence_source==None or sentence_changed==None:
        return '请输入句子'
    result = get_similarity_score(sentence_source,sentence_changed)
    return str(result)

#获得句子字词变化的得分的接口    没得问题
@app.route('/change_word_score',methods=['post'])
def change_word_score():
    sentence_source = request.form.get('sentence_source')
    sentence_changed = request.form.get('sentence_changed')
    print('原句子')
    print(sentence_source)
    print('生成的句子')
    print(sentence_changed)
    temp_list = [sentence_source, sentence_changed]
    X = vectorizer.fit_transform(temp_list).toarray()
    score = np.sqrt(np.sum(np.square(X[0] - X[1])))

    return str(round(score,4))

#词性标注的接口
@app.route('/pog_tagging', methods=['post'])
def get_pog_tagging():
    sentence = request.form.get('sentence')
    print(sentence)
    return jsonify(pog_tagging(sentence))

#获取同义词的接口
@app.route('/get_synonyms',methods=['post'])
def get_word_synonyms():
    word = request.form.get('word')
    result = {}
    for i in get_synonyms(word):
        result[i[0]] = float(i[1])
    print(result)
    return jsonify(result)

#整个句子获取同义词
@app.route('/sentence_synonyms',methods=['post'])
def get_sentence_synonyms():
    sentence = request.form.get('sentence')
    print(sentence)
    result = []
    for i in sentence.split(' '):
        while True:
            if i[-1] not in string.ascii_letters:
                i = i[:-1]
            else:
                break
        finall_list = get_synonyms(i)
        mylist = [nn[0] for nn in finall_list]
        result.append([i, mylist])
    return jsonify(result)

# #文本生成的接口
# app.route('/text_generation',methods=['post'])
# def get_text():
#     senetence = request.form.get('sentence')
#     words_number = request.form.get('word_number')
#     text = text_generation(senetence,words_number)
#     return text
#
#语法纠错的接口
#@app.route('/grammer_correct',methods=['post'])
#def grammer_correct():
#    sentence = request.form.get('sentence')
#    print(sentence)
 #   corrected_sentence = gf.correct(sentence,max_candidates=1)
  #  item = {"corrected_sentence":corrected_sentence}
   # return jsonify(item)

# @app.route('/summary_text',methods=['post'])
# def summary():
#     text = request.form.get('text')
#     max_word = request.form.get('max_words')
#     min_word = request.form.get('min_words')
#     summary_text = summarizer(text,max_length=int(max_word),min_length=int(min_word),do_sample=False)[0]
#     return jsonify(summary_text)


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8802,debug=False,use_reloader=False) #threaded=False速度变慢了好多
    #这个host：windows就一个网卡，可以不写，而liux有多个网卡，写成0:0:0可以接受任意网卡信息
    #通过访问127.0.0.1:8802/get_user，form—data里输入username参数，则可看到返回信息
import lda
import datetime
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.ticker import MultipleLocator

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

var_pickle_path = '../var_dump/index_dict_2017_20topics_2000Epochs_1534941301.var2'
index_dict = load_variavle(var_pickle_path)

# 创建停用词list(可以直接调用黑名单)
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist('../config/stopwords.txt')

# 层次抽取之后再看按月的情况如何
# ggkf_corpus_id = index_dict[2]
# party_corpus_id = index_dict[5]
# economy_corpus_id = index_dict[7]
# medicalEdu_corpus_id = index_dict[10]
corpus_id = index_dict[7]


def month_classifier(doc_keywords_file, corpus_id):
    month_docs_dict = {}
    corpus = list()
    corpus_index = list()

    # 其实也就存在12种情况而已, 如果10几万条数据信息都判断一遍运算效率就大大降低了
    for month in range(12):
        month_docs_dict[month] = list()

    with open(doc_keywords_file, 'r') as fp:
        line = fp.readline().strip()
        index = 0
        while line:
            id, date, kws = line.split(',')
            if id in corpus_id:
                kws = kws.split('|')
                date = datetime.datetime.strptime(date, "%Y-%m-%d")
                month = date.month - 1  # 月份共分为 0 - 11
                # 仅保留关键词数大于4的报道

                kwsWithoutNum = []
                for kw in kws:
                    if re.compile(r'-?[0-9]\d*').match(kw):
                        continue
                    kwsWithoutNum.append(kw)

                if len(kwsWithoutNum) > 4:
                    content = ' '.join(kwsWithoutNum)
                    corpus.append(content)
                    corpus_index.append(index)
                    month_docs_dict[month].append((id, date, content))
            line = fp.readline().strip()  # 读取下一行
            index += 1
    return month_docs_dict, corpus, corpus_index


doc_keywords_file = '/home/zhangchj/PycharmProjects/BLMYX01/res/topic_extraction/dedup_mine_2017_doc_kws2.csv'
month_docs_dict, corpus, corpus_index = month_classifier(doc_keywords_file, corpus_id)
print('finish loading month_docs_dict and corpus')

def show_docNum_monthly(month_docs_dict):
    report_num = []
    for month in range(12):
        report_num.append(len(month_docs_dict[month]))
    return report_num

# print(show_docNum_monthly(month_docs_dict)) # 观察随着月份推移相关主题文章数量的变化
report_num = show_docNum_monthly(month_docs_dict)
Month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']


for a,b in zip(Month,report_num):
    plt.text(a,b, b, ha='center', va='bottom', fontsize=10)

plt.plot(Month, report_num, '-o', color='orange', markersize=5)
plt.grid(Month, color='grey',linestyle='dashed')
plt.ylabel('Report Number')
plt.xlabel('Month')
ax = plt.gca()
# ax.xaxis.set_major_locator(MultipleLocator(3))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.yaxis.set_minor_locator(MultipleLocator(40))
# plt.xlim(0,12)
plt.ylim(0, 1000)
plt.show()


vectorizer = CountVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
print('finish handling vectorizer')

n_topics = 10 # 划分主题量
n_iter = 500 # 迭代次数

model =lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
model.fit_transform(X)
print('finish training')

#文档-主题（Document-Topic）分布 (该变量记录了每一篇文档隶属于哪个主题)
doc_topic = model.doc_topic_
print("\n shape: {}".format(doc_topic.shape))
topic_word = model.topic_word_
word = vectorizer.get_feature_names()

n = 20 # 显示关键词数量（这个是按頻数)
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(word)[np.argsort(topic_dist)][:-(n + 1):-1]
    print(u'*Topic {},{}'.format(i, ' '.join(topic_words)))

topic_dict = dict()
# 先初始化所有类别的索引
for category in range(n_topics):
    topic_dict[category] = 0

# # 如下代码用于corpus为整个完整预料库(corpus.append(' '.join(kws))在if id in corpus_id判断的外部)
# for index in corpus_index:
#     category = doc_topic[index].argmax()
#     topic_dict[category] += 1
for index in range(len(corpus_index)):
    category = doc_topic[index].argmax()
    topic_dict[category] += 1


# 如下代码负责输出各主题下对应文章数及主题权重
for item in topic_dict.items():
    print("Topic: {}, Num: {}, Weight: {:.2f}%".format(item[0], item[1], 100*item[1]/len(corpus_id)))  # 如此运算无法进行四舍五入(算了，这个误差可以忽略)




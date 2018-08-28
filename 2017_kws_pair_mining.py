import pickle
import datetime

'''
词组频率提取及统计（全部建立在能够正确分词的基础上）
# 后续可以考虑要找什么关键词组合，然后看这个关键词组合随着时间推移的数量变化情况
'''

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

# 这个需要改进一下，只根据变量内容
# 这个不需要count
def kws_pair_stats_3(topic_extracted_file, corpus_id):
    kw_pair_stats = dict()
    with open(topic_extracted_file) as fp:
        line = fp.readline()
        # counter = 0
        while line:
            parts = line.strip().split(',')
            id = parts[0]
            if id in corpus_id:
                kws = parts[2].split('|')
                kws.sort() # 这里经过sort()之后就不必再进行正反向匹配了
                for i in range(len(kws)):
                    for j in range(i+1, len(kws)):
                        for k in range(j+1, len(kws)):
                            pair = (kws[i], kws[j], kws[k])
                            if pair not in kw_pair_stats:
                                kw_pair_stats[pair] = 0
                            kw_pair_stats[pair] += 1
            line = fp.readline()
            # counter += 1

            # if counter % 5000 == 0:
            #     print(counter)
    sorted_stats = sorted(kw_pair_stats.items(), key=lambda x: x[1], reverse=True)
    return sorted_stats


def kws_pair_stats_4(topic_extracted_file, corpus_id):
    kw_pair_stats = dict()
    with open(topic_extracted_file) as fp:
        line = fp.readline()
        # counter = 0
        while line:
            parts = line.strip().split(',')
            id = parts[0]
            if id in corpus_id:
                kws = parts[2].split('|')
                kws.sort() # 这里经过sort()之后就不必再进行正反向匹配了
                for i in range(len(kws)):
                    for j in range(i+1, len(kws)):
                        for k in range(j+1, len(kws)):
                            for l in range(k + 1, len(kws)):
                                pair = (kws[i], kws[j], kws[k], kws[l])
                                if pair not in kw_pair_stats:
                                    kw_pair_stats[pair] = 0
                                kw_pair_stats[pair] += 1
            line = fp.readline()
            # counter += 1

            # if counter % 5000 == 0:
            #     print(counter)
    sorted_stats = sorted(kw_pair_stats.items(), key=lambda x: x[1], reverse=True)
    return sorted_stats

def save_kws_pair_stats_3(sorted_stats, output_path):
    # sorted_stats = sorted(kw_pair_stats.items(), key = lambda x:x[1], reverse = True)
    with open(output_path, 'w', encoding='utf-8') as wfp:
        for item in sorted_stats:
            if item[1] > 20:
                wfp.write(item[0]+','+item[1])

# 尝试使用多进程，先通过文件拆分将topic_extracted_file(或者是通过切片将那个id)拆为多份，然后再进行匹配
# 该函数默认target_pair仅仅只是1个元组，而不是一个数列
def pair_monthly_stats(topic_extracted_file, corpus_id, target_pair):
    stat_dict = dict()
    print('handling keyword pair')
    for month in range(12):
        stat_dict[month] = 0

    with open(topic_extracted_file) as fp:
        line = fp.readline().strip()
        counter = 0
        while line:
            parts = line.split(',')
            id = parts[0]
            if id in corpus_id:
                kws = parts[2].split('|')
                if (target_pair[0] not in kws) or (target_pair[1] not in kws) or (target_pair[2] not in kws):
                    line = fp.readline().strip() # 如果缺了这一行直接continue就会陷入死循环了
                    continue
                date = datetime.datetime.strptime(parts[1], "%Y-%m-%d")
                stat_dict[date.month-1] += 1
            line = fp.readline().strip()
            counter += 1

            if counter % 10000 == 0:
                print(counter)
    return stat_dict

def pair2_monthly_stats(topic_extracted_file, corpus_id, target_pair):
    stat_dict = dict()
    print('handling keyword pair')
    for month in range(12):
        stat_dict[month] = 0

    with open(topic_extracted_file) as fp:
        line = fp.readline().strip()
        counter = 0
        while line:
            parts = line.split(',')
            id = parts[0]
            if id in corpus_id:
                kws = parts[2].split('|')
                if (target_pair[0] not in kws) or (target_pair[1] not in kws):
                    line = fp.readline().strip() # 如果缺了这一行直接continue就会陷入死循环了
                    continue
                date = datetime.datetime.strptime(parts[1], "%Y-%m-%d")
                stat_dict[date.month-1] += 1
            line = fp.readline().strip()
            counter += 1

            if counter % 10000 == 0:
                print(counter)
    return stat_dict

def kw_monthly_stats(topic_extracted_file, corpus_id, target_kw):
    stat_dict = dict()
    print('handling keyword')
    for month in range(12):
        stat_dict[month] = 0

    with open(topic_extracted_file) as fp:
        line = fp.readline().strip()
        counter = 0
        while line:
            parts = line.split(',')
            id = parts[0]
            if id in corpus_id:
                kws = parts[2].split('|')
                if target_kw not in kws:
                    line = fp.readline().strip() # 如果缺了这一行直接continue就会陷入死循环了
                    continue
                date = datetime.datetime.strptime(parts[1], "%Y-%m-%d")
                stat_dict[date.month-1] += 1
            line = fp.readline().strip()
            counter += 1

            if counter % 10000 == 0:
                print(counter)
    return stat_dict


if __name__ == '__main__':
    var_pickle_path = './var_dump/index_dict_2017_20topics_2000Epochs_1534941301.var2'
    index_dict = load_variavle(var_pickle_path)
    print('finish loading index_dict')
    topic_extracted_file = '/home/zhangchj/PycharmProjects/BLMYX01/res/topic_extraction/dedup_mine_2017_doc_kws2.csv'
    # 它的那个分词效果不行，试着自己写一个
    # num = 15
    # for topic, corpus_ids in index_dict.items():
    #     # sorted_stats = kws_pair_stats_3(topic_extracted_file, corpus_ids) # 字典：关键词：頻数
    #     sorted_stats = kws_pair_stats_4(topic_extracted_file, corpus_ids) # 字典：关键词：頻数
    #     counter = 0
    #     print('Topic:{}'.format(topic))
    #     for pair, frequency in sorted_stats:
    #         print(pair, frequency)
    #         counter += 1
    #         if counter >= num:
    #             # 这个break是跳出最里层循环
    #             break


    # output_path = './pairMining/ggkf_corpus_id.txt'
    # save_kws_pair_stats_3(sorted_stats, output_path)
    #
    # # 观察在某个文档集合里边包含某个词组的每个月的数量变化情况
    # target_pair = ('一带一路', '合作', '高峰论坛')
    target_pair = ('习近平', '合作')

    corpus_id = index_dict[1]+index_dict[7]+index_dict[10]+index_dict[5]+index_dict[16]+index_dict[15]+index_dict[14]+index_dict[12]+index_dict[11]+index_dict[9]
    stat_dict = pair2_monthly_stats(topic_extracted_file, corpus_id, target_pair)
    # stat_dict = kw_monthly_stats(topic_extracted_file, corpus_id, target_kw)
    for month, num in stat_dict.items():
        print('Month: {}, reportNum: {}'.format(month+1, num))


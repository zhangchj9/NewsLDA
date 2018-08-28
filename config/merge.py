terms = []

# with open('./keyword_blacklist.txt', encoding='utf-8') as fp, open('./keyword_blacklist_merged.txt', 'a', encoding='utf-8') as wfp:
with open('./userdict_merged3.txt', encoding='utf-8') as fp, open('./userdict_merged4.txt', 'a', encoding='utf-8') as wfp:
# with open('./userdict.txt', encoding='utf-8') as fp, open('./userdict_merged.txt', 'a', encoding='utf-8') as wfp:
    for line in fp.readlines():
        # parts = line.strip().split(',')
        # oneline = list()
        # oneline.append(parts[0])
        # kws = parts[1].split('|')
        # for kw in kws:
        #     oneline.append(kw)
        if line.strip().startswith('@') and line.strip() != '':
            line = line.split('@')[1]

        terms.append(line.strip().split('\t')[0])

    for term in set(terms):
        wfp.write(term.strip() + '\n')

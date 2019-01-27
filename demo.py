import re
import pickle
import itertools
import sys

print('Loading the corpus ...')
corpus = []
with open('data/corpus_subset.txt', 'r') as f:
    for line in f:
        corpus.append(line.strip('\n').split('\t'))

print('Loading the index ...')
with open('data/mp_index', 'rb') as f:
    mp_index = pickle.load(f)
with open('data/entity_index', 'rb') as f:
    entity_index = pickle.load(f)
with open('data/word_index', 'rb') as f:
    word_index = pickle.load(f)
with open('data/entity2id', 'rb') as f:
    entity2id = pickle.load(f)
with open('data/entity_tfidf', 'rb') as f:
    entity_tfidf = pickle.load(f)
with open('data/entity_names', 'rb') as f:
    entity_names = pickle.load(f)
with open('data/word_tfidf', 'rb') as f:
    word_tfidf = pickle.load(f)
with open('data/word_names', 'rb') as f:
    word_names = pickle.load(f)
# with open('data/mp2score', 'rb') as f:
#     mp2score = pickle.load(f)
with open('data/type2entity', 'rb') as f:
    type2entity = pickle.load(f)


def word_match(query_words, word_index):
    sets = []
    for word in query_words:
        if word in word_index:
            sets.append(word_index[word])
    if sets == []:
        return set()
    # ret = set.intersection(*sets)
    ret = set.union(*sets)
    return ret

def entity_match(query_entities, entity_index):
    sets = []
    for entity in query_entities:
        if entity in entity_index:
            sets.append(entity_index[entity])
    if sets == []:
        return set()
    # ret = set.intersection(*sets)
    ret = set.union(*sets)
    return ret

def mp_match(query_entities, mp_index):
    ret = set()
    ret2scores = {}
    queries = set(query_entities)
    for key in mp_index:
        keys = set(key.split('\t'))
        if len(queries & keys) > 0:
            score = len(queries & keys)/len(queries | keys) # Jaccard similarity
            ret = ret | mp_index[key]
            for sent in mp_index[key]:
                if sent not in ret2scores:
                    ret2scores[sent] = []
                ret2scores[sent].append(score)
    return ret, ret2scores

def claimRank(query, mp_index, entity_index, word_index, entity2id, word_names, word_tfidf, entity_names, entity_tfidf, weight):
    # tf-idf score of word match
    # tf-idf score of entity match
    # score of pattern match (exact match, over-match, partial match)
    query_words = [w for word in query for w in word.split('_')]
    query_entities = [entity2id[word] for word in query if word in entity2id]

    word_set = word_match(query_words, word_index)
    entity_set = entity_match(query_entities, entity_index)
    combined_set = word_set | entity_set
    mp_set, mp_scores = mp_match(query_entities, mp_index)
    mp_set = mp_set & combined_set
    # print(word_set, entity_set, mp_set, word_set.issubset(entity_set))


    key2scores = {}
    for key in combined_set:
        key2scores[key] = [0,0,0]
        row = key
        word_exist = []
        entity_exist = []

        # Score of words
        if key in word_set:
            for word in query_words:
                if word not in word_names:
                    word_exist.append(False)
                    continue
                col = word_names.index(word)
                key2scores[key][0] += word_tfidf[row, col]
                word_exist.append(word_tfidf[row, col]>0)
            key2scores[key][0] = key2scores[key][0]/len(query_words)

        # Score of entities
        if key in entity_set:
            for entity in query_entities:
                col = entity_names.index(entity)
                key2scores[key][1] += entity_tfidf[row, col]
                entity_exist.append(entity_tfidf[row, col]>0)
            key2scores[key][1] = key2scores[key][1]/len(query_entities)

        # Score of mps
        if key in mp_set:
             key2scores[key][2] = sum(mp_scores[key])/len(mp_scores[key])

        # Check the coverage of entity and word
        flag = 0
        for word in query:
            exist1 = False
            exist2 = False
            if word_exist != []:
                exist1 = min(word_exist[:len(word.split('_'))])
                word_exist = word_exist[len(word.split('_')):]
            if entity_exist != []:
                if word in query_entities:
                    exist2 = entity_exist.pop(0)
            exist = exist1 | exist2
            if exist == False:
                flag = 1
                break

        if flag == 1:
            del key2scores[key]

    key2score = [(key, key2scores[key], sum([x*y for x,y in zip(key2scores[key],weight)])/3) for key in key2scores]
    return sorted(key2score, key=lambda x: x[2], reverse=True)

def output(ranklist):
    return [[corpus[i[0]][0], corpus[i[0]][1], i[1][0], i[1][1], i[1][2], i[2]] for i in ranklist]

def claimMiner(query, mp_index, entity_index, word_index, entity2id, word_names, word_tfidf, entity_names, entity_tfidf, weight):
    # Example queries
    query = [re.sub('[^0-9a-zA-Z]+', '_', q.lower()) for q in query] # Query processing

    # mp_list = rankList(query, mp_index, 'mp')
    # entity_list = rankList(query, entity_index, 'entity')
    # word_list = rankList(query, word_index, 'word')
    # flist = mergeList([mp_list, entity_list, word_list])
    # fout = output(flist)
    ranklist = claimRank(query, mp_index, entity_index, word_index, entity2id, word_names, word_tfidf, entity_names, entity_tfidf, weight)
    ret = output(ranklist)
    return ret

# Main function takes two forms of input: (1) arsenic; cardio_renal_toxicity, (arsenic; $DISEASE)
def demo(input_query, mp_index, entity_index, word_index, entity2id, word_names, word_tfidf, entity_names, entity_tfidf, weight = [1, 1, 1]):
    queries = []
    for q in input_query:
        if q[0] == '$':
            if q[1:] not in type2entity.keys():
                print (i + ' is not a correct entity type: ' + str(list(type2entity.keys())))
                return []
            else:
                queries.append(type2entity[q[1:]])
        else:
            queries.append([q])

    new_queries = itertools.product(*queries)

    fouts = []
    selected_queries = []
    for new_query in new_queries:
        fout = claimMiner(new_query, mp_index, entity_index, word_index, entity2id, word_names, word_tfidf, entity_names, entity_tfidf, weight)
        if len(fout) == 0:
            continue
        selected_queries.append(new_query)
        fouts.append(fout)
    return sorted(list(zip(selected_queries, fouts)), key=lambda x: len(x[1]), reverse=True)


if __name__ == '__main__':
    print('Input the query: ')
    for line in sys.stdin:
        if line == '\n':
            continue
        elif line.strip('\n').lower() == 'exit':
            break

        input_query = line.strip('\n').split('; ')
        input_query = [re.sub('[^0-9a-zA-Z]+', ' ', q.lower()) for q in input_query]
        demo_outs = demo(input_query, mp_index, entity_index, word_index, entity2id, word_names, word_tfidf, entity_names, entity_tfidf, [1,0,0])

        # Check the output
        print('User input: ', input_query)
        print('Number of generated queries: ', len(demo_outs))
        for (query, fout) in demo_outs:
            print('------------------------\n')
            print('Input query: ', query)
            print('Number of output claims: ', len(fout))
            print('--------')
            N = min(len(fout), 100)
            for i in range(len(fout)):
                print(i+1, (fout[i]))

        print('\n')
        print('Input the query: ')

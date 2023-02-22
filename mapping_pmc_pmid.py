import pickle
filelist = "/Users/albertoberenguerpastor/Downloads/oa_file_list.txt"

pmids = dict()
with open(filelist, 'r') as f:
    for line in f:
        info = line.split('\t')
        if len(info) <= 3:
            continue
        else:
            pmid = info[3]
            pmc = info[2]
            if pmid.startswith('PMID:'):
                pmid = pmid[5:]
                pmids[pmid] = pmc

with open('./mapping.pickle', 'wb') as handle:
    pickle.dump(pmids, handle, protocol=pickle.HIGHEST_PROTOCOL)

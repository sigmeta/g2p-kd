import os


'''
Count all top k results and get the voting result
'''


path="result/vote"
files=os.listdir(path)
dicts=[]    #Record the results of each row for final sorting
decay=0.2	#The number of weight reductions. Top1 weight is 1, then top2 weight is 1-decay
for file in files:
    with open(os.path.join(path,file)) as f:
        txt=f.read()
        lines=txt.strip().split('\n')
        if not dicts:
            dicts=[{} for _ in range(len(lines))]
        for i,line in enumerate(lines):
            topk=line.strip().split('\t')
            for j,r in enumerate(topk):
                if r in dicts[i]:
                    dicts[i][r]+=(1-j*decay)
                else:
                    dicts[i][r] = (1 - j * decay)
res=[]
for i,d in enumerate(dicts):
    max=0
    mkey=''
    for k in d:
        if d[k]>max:
            max=d[k]
            mkey=k
    res.append(mkey)

with open("result/vote.txt",'w') as f:
    f.write('\n'.join(res))


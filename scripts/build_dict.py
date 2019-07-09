import argparse
import os


def build_dictionary(args):
    files=args.src.split(',')
    # Get all text
    txt_list=[]
    for file in files:
        with open(file,encoding='utf8') as f:
            txt_list+=f.read().strip().replace(' ','').split('\n')
    uni_dict={}
    bi_dict={}
    tri_dict={}
    for t in txt_list:
        for i in range(len(t)):
            if t[i] in uni_dict:
                uni_dict[t[i]]+=1
            else:
                uni_dict[t[i]]=1
        for i in range(len(t)-1):
            if t[i:i+2] in bi_dict:
                bi_dict[t[i:i+2]]+=1
            else:
                bi_dict[t[i:i+2]]=1
        for i in range(len(t)-2):
            if t[i:i+3] in tri_dict:
                tri_dict[t[i:i+3]]+=1
            else:
                tri_dict[t[i:i+3]]=1
    with open(os.path.join(args.tgt,"uni.dict.txt"),'w',encoding='utf8') as f:
        items=sorted(uni_dict.items(), key=lambda d: d[1], reverse=True)
        f.write("\n".join([k+' '+str(v) for k,v in items]))
    with open(os.path.join(args.tgt,"bi.dict.txt"),'w',encoding='utf8') as f:
        items=sorted(bi_dict.items(), key=lambda d: d[1], reverse=True)
        f.write("\n".join([k+' '+str(v) for k,v in items]))
    with open(os.path.join(args.tgt,"tri.dict.txt"),'w',encoding='utf8') as f:
        items=sorted(tri_dict.items(), key=lambda d: d[1], reverse=True)
        f.write("\n".join([k+' '+str(v) for k,v in items]))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',default="")
    parser.add_argument('--tgt',default="")
    args = parser.parse_args()
    build_dictionary(args)



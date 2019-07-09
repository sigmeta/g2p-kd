import argparse
import os
import numpy as np
from numpy import linalg

def get_dictionary(args):
    with open(os.path.join(args.dict, "uni.dict.txt"),encoding='utf8') as f:
        uni_dict={line.split()[0]:i+1 for i,line in enumerate(f)}
    with open(os.path.join(args.dict, "bi.dict.txt"),encoding='utf8') as f:
        bi_dict={line.split()[0]:i+1 for i,line in enumerate(f)}
    with open(os.path.join(args.dict, "tri.dict.txt"),encoding='utf8') as f:
        tri_dict={line.split()[0]:i+1 for i,line in enumerate(f)}
    return uni_dict,bi_dict,tri_dict


def get_vec_ref(file,uni_dict,bi_dict,tri_dict):
    with open(file,encoding='utf8') as f:
        tlist=f.read().strip().replace(' ','').split('\n')
    uni_array=np.zeros((1,len(uni_dict)+1))   # 0 - unknown character
    bi_array=np.zeros((1,len(bi_dict)+1))
    tri_array=np.zeros((1,len(tri_dict)+1))
    for t in tlist:
        for i in range(len(t)):
            if t[i] in uni_dict:
                uni_array[0,uni_dict[t[i]]]+=1
            else:
                uni_array[0,0]=1
        for i in range(len(t)-1):
            if t[i:i+2] in bi_dict:
                bi_array[0,bi_dict[t[i:i+2]]]+=1
            else:
                bi_array[0,0]=1
        for i in range(len(t)-2):
            if t[i:i+3] in tri_dict:
                tri_array[0,tri_dict[t[i:i+3]]]+=1
            else:
                tri_array[0,0]=1
    uni_array=uni_array/np.sum(uni_array)
    bi_array=bi_array/np.sum(bi_array)
    tri_array=tri_array/np.sum(tri_array)
    return uni_array,bi_array,tri_array


def get_vec_inf(t,uni_dict,bi_dict,tri_dict):
    uni_array=np.zeros((1,len(uni_dict)+1))   # 0 - unknown character
    bi_array=np.zeros((1,len(bi_dict)+1))
    tri_array=np.zeros((1,len(tri_dict)+1))
    for i in range(len(t)):
        if t[i] in uni_dict:
            uni_array[0,uni_dict[t[i]]]+=1
        else:
            uni_array[0,0]=1
    for i in range(len(t)-1):
        if t[i:i+2] in bi_dict:
            bi_array[0,bi_dict[t[i:i+2]]]+=1
        else:
            bi_array[0,0]=1
    for i in range(len(t)-2):
        if t[i:i+3] in tri_dict:
            tri_array[0,tri_dict[t[i:i+3]]]+=1
        else:
            tri_array[0,0]=1
    uni_array=uni_array/np.sum(uni_array,axis=1,keepdims=True)
    bi_array=bi_array/(np.sum(bi_array,axis=1,keepdims=True)+1e-10)
    tri_array=tri_array/(np.sum(tri_array,axis=1,keepdims=True)+1e-10)
    return uni_array,bi_array,tri_array


def get_similarity(a1,a2):
    # cosine
    #return np.matmul(a1,a2.T)/(linalg.norm(a1)*linalg.norm(a2))
    # distance
    return 1/(1+np.sum(np.abs(a1-a2),axis=1,keepdims=True))
    # dot
    #return np.matmul(a1, a2.T)

def get_topk(args):
    uni_dict,bi_dict,tri_dict=get_dictionary(args)
    with open(args.inference,encoding='utf8') as f:
        tlist=f.read().strip().replace(' ','').split('\n')
    ru,rb,rt=get_vec_ref(args.reference,uni_dict,bi_dict,tri_dict)
    print("got reference")
    iu,ib,it=get_vec_inf(args.inference,uni_dict,bi_dict,tri_dict)
    print("got inference")
    s1=get_similarity(iu,ru)
    s2=get_similarity(ib,rb)
    s3=get_similarity(it,rt)
    sml=(s1+s2+s3)/3
    print("got similarity")
    slist=zip(tlist,sml.squeeze().tolist())
    slist=sorted(slist, key=lambda k:k[1], reverse=True)
    with open(args.tgt,'w',encoding='utf8') as f:
        f.write('\n'.join([' '.join(s[0]) for s in slist[:args.top]]))



def similar(args):
    uni_dict, bi_dict, tri_dict = get_dictionary(args)
    au1, ab1, at1 = get_vec_ref(args.reference,uni_dict,bi_dict,tri_dict)
    au2, ab2, at2 = get_vec_ref(args.inference, uni_dict, bi_dict, tri_dict)
    s1=get_similarity(au1,au2)
    s2=get_similarity(ab1,ab2)
    s3=get_similarity(at1,at2)
    print("similarity:", (s1+s2+s3)/3,s1,s2,s3)


def get_best2(args): #greedy
    uni_dict,bi_dict,tri_dict=get_dictionary(args)
    with open(args.inference,encoding='utf8') as f:
        tlist=f.read().strip().replace(' ','').split('\n')
    ru,rb,rt=get_vec_ref(args.reference,uni_dict,bi_dict,tri_dict)
    print("got reference")
    best_score = 0
    res_list = []
    bu = np.zeros((1, 1))
    bb = np.zeros((1, 1))
    bt = np.zeros((1, 1))
    for i in range(len(tlist)):
        if i % 10000 == 0:
            print(i, best_score, len(res_list))
        iu, ib, it = get_vec_inf(tlist[i], uni_dict, bi_dict, tri_dict)
        res_list.append(i)
        nu = (bu * (len(res_list) - 1) + iu) / len(res_list)
        nb = (bb * (len(res_list) - 1) + ib) / len(res_list)
        nt = (bt * (len(res_list) - 1) + it) / len(res_list)
        s1 = get_similarity(nu, ru)
        s2 = get_similarity(nb, rb)
        s3 = get_similarity(nt, rt)
        sml = (s1 + s2 + s3) / 3
        sml = sml.squeeze()
        if sml >= best_score:
            best_score = sml
            bu, bb, bt = nu, nb, nt
        else:
            res_list.pop()
    for i in range(len(tlist)):
        if i % 10000 == 0:
            print(i, best_score, len(res_list))
        if i in res_list:
            continue
        iu, ib, it = get_vec_inf(tlist[i], uni_dict, bi_dict, tri_dict)
        res_list.append(i)
        nu = (bu * (len(res_list) - 1) + iu) / len(res_list)
        nb = (bb * (len(res_list) - 1) + ib) / len(res_list)
        nt = (bt * (len(res_list) - 1) + it) / len(res_list)
        s1 = get_similarity(nu, ru)
        s2 = get_similarity(nb, rb)
        s3 = get_similarity(nt, rt)
        sml = (s1 + s2 + s3) / 3
        sml = sml.squeeze()
        if sml >= best_score or sml > 0.9:
            best_score = sml
            bu, bb, bt = nu, nb, nt
        else:
            res_list.pop()
    slist = []
    for i in sorted(res_list):
        slist.append(tlist[i])
    with open(args.tgt,'w',encoding='utf8') as f:
        f.write('\n'.join([' '.join(s) for s in slist]))


def get_best(args): #greedy
    uni_dict,bi_dict,tri_dict=get_dictionary(args)
    with open(args.inference,encoding='utf8') as f:
        tlist=f.read().strip().replace(' ','').split('\n')
    ru,rb,rt=get_vec_ref(args.reference,uni_dict,bi_dict,tri_dict)
    print("got reference")

    #print("got inference")
    best_score=0
    res_list=[]
    bu=np.zeros((1,1))
    bb=np.zeros((1,1))
    bt=np.zeros((1,1))
    for i in range(len(tlist)):
        if i%10000==0:
            print(i,best_score,len(res_list))
        iu, ib, it = get_vec_inf(tlist[i], uni_dict, bi_dict, tri_dict)
        res_list.append(i)
        nu=(bu*(len(res_list)-1)+iu)/len(res_list)
        nb= (bb*(len(res_list)-1)+ib)/len(res_list)
        nt = (bt*(len(res_list)-1)+it)/len(res_list)
        s1 = get_similarity(nu, ru)
        s2 = get_similarity(nb, rb)
        s3 = get_similarity(nt, rt)
        sml = (s1 + s2 + s3) / 3
        sml=sml.squeeze()
        if sml>=best_score:
            best_score=sml
            bu,bb,bt=nu,nb,nt
        else:
            res_list.pop()
    for i in range(len(tlist)):
        if i%10000==0:
            print(i,best_score,len(res_list))
        if i in res_list:
            continue
        iu, ib, it = get_vec_inf(tlist[i], uni_dict, bi_dict, tri_dict)
        res_list.append(i)
        nu=(bu*(len(res_list)-1)+iu)/len(res_list)
        nb= (bb*(len(res_list)-1)+ib)/len(res_list)
        nt = (bt*(len(res_list)-1)+it)/len(res_list)
        s1 = get_similarity(nu, ru)
        s2 = get_similarity(nb, rb)
        s3 = get_similarity(nt, rt)
        sml = (s1 + s2 + s3) / 3
        sml=sml.squeeze()
        if sml>=best_score or sml>0.9:
            best_score=sml
            bu,bb,bt=nu,nb,nt
        else:
            res_list.pop()
    slist=[]
    for i in sorted(res_list):
        slist.append(tlist[i])
    with open(args.tgt,'w',encoding='utf8') as f:
        f.write('\n'.join([' '.join(s) for s in slist]))
    with open(args.tgt+'.list','w',encoding='utf8') as f:
        f.write('\n'.join([str(s) for s in sorted(res_list)]))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task',choices=["similar","topk","best"])
    parser.add_argument('-i','--inference',default="")
    parser.add_argument('-r','--reference',default="")
    parser.add_argument('--tgt',default="path to save topk result")
    parser.add_argument('--dict')
    parser.add_argument('--top',default=100000, type=int)


    args = parser.parse_args()
    if args.task=="similar":
        similar(args)
    elif args.task=="topk":
        get_topk(args)
    elif args.task=="best":
        get_best(args)


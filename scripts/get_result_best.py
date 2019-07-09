import os

for i in range(53,77):
    print("t",i)
    with open("../output/best/t"+str(i)) as f:
        print(f.read())
    try:
        flist = os.listdir("/hdfs/sdrgvc/xuta/t-hasu/checkpoints/t" + str(i))
    except:
        print("deleted");continue
    nlist = []
    for f in flist:
        if '_' not in f:
            try:
                nlist.append(int(f[10:-3]))
            except Exception as e:
                print(e)
    n = max(nlist)
    print(n)


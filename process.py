import subprocess
import os
import time
while (True):
    for i in []+[num for num in range(172,174)]:
        print(i)
        try:
            flist = os.listdir("/hdfs/sdrgvc/xuta/t-hasu/checkpoints/stu" + str(i))
        except:
            print("deleted");continue
        nlist = []
        for f in flist:
            if '_' not in f:
                try:
                    nlist.append(int(f[10:-3]))
                except Exception as e:
                    print(e)
        if not nlist: continue
        n = max(nlist)
        print(n)
       # time.sleep(0.2)
        stat = subprocess.call(
            "python cal.py data-bin/lts  --path /hdfs/sdrgvc/xuta/t-hasu/checkpoints/stu" + str(
                i) + "/checkpoint" + str(n) + ".pt  --beam 10  --quiet", shell=True)
        if stat:
            stat = subprocess.call(
                "python cal.py data-bin/lts  --path /hdfs/sdrgvc/xuta/t-hasu/checkpoints/stu" + str(
                    i) + "/checkpoint" + str(n - 10) + ".pt  --beam 10  --quiet", shell=True)
    time.sleep(30)

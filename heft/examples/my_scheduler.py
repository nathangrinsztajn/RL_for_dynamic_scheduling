from heft.core_heft import (wbar, cbar, ranku, schedule, Event, start_time,
                            makespan, endtime, insert_recvs, insert_sends, insert_sendrecvs, recvs,
                            sends)

"""
This is a simple script to use the HEFT function provided based on the example given in the original HEFT paper.
You have to define the DAG, compcost function and commcost funtion.

Each task/job is numbered 1 to 10
Each processor/agent is named 'a', 'b' and 'c'

Output expected:
Ranking:
[10, 8, 7, 9, 6, 5, 2, 4, 3, 1]
Schedule:
('a', [Event(job=2, start=27, end=40), Event(job=8, start=57, end=62)])
('b', [Event(job=4, start=18, end=26), Event(job=6, start=26, end=42), Event(job=9, start=56, end=68), Event(job=10, start=73, end=80)])
('c', [Event(job=1, start=0, end=9), Event(job=3, start=9, end=28), Event(job=5, start=28, end=38), Event(job=7, start=38, end=49)])
{1: 'c', 2: 'a', 3: 'c', 4: 'b', 5: 'c', 6: 'b', 7: 'c', 8: 'a', 9: 'b', 10: 'b'}
"""


dag={1:(2,3,4,5,6),
     2:(8,9),
     3:(7,),
     4:(8,9),
     5:(9,),
     6:(8,),
     7:(10,),
     8:(10,),
     9:(10,),
     10:()}


def compcost(job, agent):
    if(job==1):
        if(agent=='a'):
            return 14
        elif(agent=='b'):
            return 16
        else:
            return 9

    if(job==2):
        if(agent=='a'):
            return 13
        elif(agent=='b'):
            return 19
        else:
            return 18
    if(job==3):
        if(agent=='a'):
            return 11
        elif(agent=='b'):
            return 13
        else:
            return 19
    if(job==4):
        if(agent=='a'):
            return 13
        elif(agent=='b'):
            return 8
        else:
            return 17
    if(job==5):
        if(agent=='a'):
            return 12
        elif(agent=='b'):
            return 13
        else:
            return 10
    if(job==6):
        if(agent=='a'):
            return 13
        elif(agent=='b'):
            return 16
        else:
            return 9
    if(job==7):
        if(agent=='a'):
            return 7
        elif(agent=='b'):
            return 15
        else:
            return 11
    if(job==8):
        if(agent=='a'):
            return 5
        elif(agent=='b'):
            return 11
        else:
            return 14
    if(job==9):
        if(agent=='a'):
            return 18
        elif(agent=='b'):
            return 12
        else:
            return 20
    if(job==10):
        if(agent=='a'):
            return 21
        elif(agent=='b'):
            return 7
        else:
            return 16



def commcost(ni, nj, A, B):

    if(A==B):
        return 0
    else:
        if(ni==1 and nj==2):
            return 18
        if(ni==1 and nj==3):
            return 12
        if(ni==1 and nj==4):
            return 9
        if(ni==1 and nj==5):
            return 11
        if(ni==1 and nj==6):
            return 14
        if(ni==2 and nj==8):
            return 19
        if(ni==2 and nj==9):
            return 16
        if(ni==3 and nj==7):
            return 23
        if(ni==4 and nj==8):
            return 27
        if(ni==4 and nj==9):
            return 23
        if(ni==5 and nj==9):
            return 13
        if(ni==6 and nj==8):
            return 15
        if(ni==7 and nj==10):
            return 17
        if(ni==8 and nj==10):
            return 11
        if(ni==9 and nj==10):
            return 13
        else:
            return 0

orders, jobson = schedule(dag, 'abc', compcost, commcost)
for eachP in sorted(orders):
    print(eachP,orders[eachP])
print(jobson)

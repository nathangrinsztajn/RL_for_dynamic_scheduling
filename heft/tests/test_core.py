from heft.core_heft import (wbar, cbar, ranku, schedule, Event, start_time,
                            makespan, endtime, insert_recvs, insert_sends, insert_sendrecvs, recvs,
                            sends)
from functools import partial
from heft.util import reverse_dict
from heft.util import find_job_event

def compcost(job, agent):
    if agent.islower():
        return job
    if agent.isupper():
        return job * 2

def commcost(ni, nj, A, B):
    if A == B:
        return 0
    if A.islower() == B.islower():
        return 3
    else:
        return 6

def zero_commcost(ni, nj, A, B):
    """A free commmunication cost function, useful in some tests for simplicity
    """
    return 0

dag = {3: (5,),
       4: (6,),
       5: (7,),
       6: (7,),
       7: (8, 9)}

def recv(fromagent, toagent, fromjob, tojob):
    return ('recv', fromjob, tojob, fromagent, toagent)

def send(fromagent, toagent, fromjob, tojob):
    return ('send', fromjob, tojob, fromagent, toagent)


def test_wbar():
    assert wbar(1, 'abc', compcost) == 1
    assert wbar(1, 'ABC', compcost) == 2

def test_cbar():
    assert cbar(1, 2, 'abc', commcost) == 3
    assert cbar(1, 2, 'Abc', commcost) == (6 + 6 + 3) / 3.

def test_ranku():
    rank = partial(ranku, agents='abc', commcost=commcost, compcost=compcost,
            succ=dag)
    w = partial(wbar, agents='abc', compcost=compcost)
    c = partial(cbar, agents='abc', commcost=commcost)
    assert isinstance(rank(6), (int, float))
    assert rank(8) == w(8)
    assert rank(7) == w(7) + c(7, 9) + rank(9)
    assert sorted((3,4,5,6,7,8,9), key=rank) == [4, 3, 6, 5, 7, 9, 8][::-1]

    d = {3: ()}
    rank = partial(ranku, agents='abc', commcost=commcost, compcost=compcost,
            succ=d)
    assert rank(3) == compcost(3, 'a')


def test_earliest_finish_time():
    orders = {'a': [Event(2, 0, 3)], 'b': []}
    jobson = {2: 'a'}
    prec = {3: (2,)}
    assert start_time(3, orders, jobson, prec, commcost, compcost, 'a') == 3
    assert start_time(3, orders, jobson, prec, commcost, compcost, 'b') == 3 + 3

def test_schedule():
    orders, jobson = schedule(dag, 'abc', compcost, commcost)
    a = jobson[4]
    b = jobson[3]
    c = (set('abc') - set((a, b))).pop()
    print a, b, c
    print orders
    assert orders == {a: [Event(4, 0, 4), Event(6, 4, 10),
                          Event(7, 11, 18), Event(9, 18, 27)],
                      b: [Event(3, 0, 3), Event(5, 3, 8), Event(8, 21, 29)],
                      c: []}

def test_makespan():
    assert makespan({'a': [Event(0, 0, 1), Event(1, 2, 3)],
                     'b': [Event(2, 3, 4)]}) == 4

def test_endtime():
    events = [Event(0, 1, 2), Event(1, 2, 3), Event(2, 3, 4)]
    assert endtime(0, events) == 2
    assert endtime(1, events) == 3

def test_recvs():
    jobson = {1: 'a', 2: 'b', 3: 'a'}
    prec = {3: (1, 2)}
    assert recvs(1, jobson, prec, recv) == []
    assert recvs(3, jobson, prec, recv) == [recv('b', 'a', 2, 3)]

def test_sends():
    jobson = {1: 'a', 2: 'b', 3: 'a'}
    succ = {1: (3, 2)}
    assert sends(1, jobson, succ, send) == [send('a', 'b', 1, 2)]
    assert sends(3, jobson, succ, send) == []

def test_insert_recvs():
    jobson = {1: 'a', 2: 'b', 3: 'a'}
    prec = {3: (1, 2)}
    aorder = [Event(1, 0, 2), Event(3, 3, 5)]
    result = insert_recvs(aorder, jobson, prec, recv)
    assert result[0] == aorder[0]
    assert result[-1] == aorder[-1]
    assert result[1].job == recv('b', 'a', 2, 3)
    assert jobson[recv('b', 'a', 2, 3)] == 'a'

    assert insert_recvs([], jobson, prec, recv) == []

def test_insert_sends():
    jobson = {1: 'a', 2: 'b', 3: 'a'}
    succ = {1: (3, 2)}
    aorder = [Event(1, 0, 2), Event(3, 3, 5)]
    result = insert_sends(aorder, jobson, succ, send)
    assert result[0] == aorder[0]
    assert result[-1] == aorder[-1]
    assert result[1].job == send('a', 'b', 1, 2)
    assert jobson[send('a', 'b', 1, 2)] == 'a'

    assert insert_sends([], jobson, succ, send) == []

def test_insert_sendrecvs():
    prec = {3: (1, 2),
            2: (1,)}
    succ = reverse_dict(prec)
    jobson = {1: 'a', 2: 'b', 3: 'a'}
    orders = {'a': [Event(1, 0, 1), Event(3, 4, 8)],
              'b': [Event(2, 2, 3)]}

    neworders, newjobson = insert_sendrecvs(orders, jobson, succ, send, recv)
    print neworders
    print newjobson
    assert Event(send('a', 'b', 1, 2), 1, 1) in neworders['a']
    assert Event(recv('a', 'b', 1, 2), 2, 2) in neworders['b']
    assert all(e in neworders[agent] for agent in orders
                                     for e in orders[agent])

def test_one_agent():
    orders, jobson = schedule(dag, ['A'], compcost, commcost)
    assert orders.keys() == ['A']
    assert set(e.job for e in orders['A']) == set((3,4,5,6,7,8,9))
    assert jobson == {i: 'A' for i in (3,4,5,6,7,8,9)}


def test_task_insertion_at_start():
    dag = {9: (10, 11), # some very high rank tasks which all depend on one
           10: (),      # initial task.
           11: (),

           1: (), # one low rank but cheap tasks
          }

    orders, _ = schedule(dag, 'ab', compcost, zero_commcost)

    # The cheap task 1 should be done on the spare processor while we are
    # waiting for task 9 to finish on the first one.
    assert find_job_event(1, orders).end == 1


def test_task_insertion_in_middle():
    dag = {9: (10, 11), # some very high rank tasks which all depend on one
           10: (),      # initial task.
           11: (),

           1: (), # multiple low rank but cheap tasks
           2: (),
          }

    orders, _ = schedule(dag, 'ab', compcost, zero_commcost)

    print orders
    print find_job_event(2, orders)

    # Both of the cheap tasks (1 and 2) should be done on the second
    # processor while we are waiting for task 9 to finish on the first one
    assert find_job_event(1, orders).end == 3
    assert find_job_event(2, orders).end == 2

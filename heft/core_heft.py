
"""
Heterogeneous Earliest Finish Time -- A static scheduling heuristic

      Performance-effective and low-complexity task scheduling
                    for heterogeneous computing
                                by
             Topcuoglu, Haluk; Hariri, Salim Wu, M
     IEEE Transactions on Parallel and Distributed Systems 2002


Cast of Characters:

job - the job to be allocated
orders - dict {agent: [jobs-run-on-agent-in-order]}
jobson - dict {job: agent-on-which-job-is-run}
prec - dict {job: (jobs which directly precede job)}
succ - dict {job: (jobs which directly succeed job)}
compcost - function :: job, agent -> time to compute job on agent
commcost - function :: job, job, agent, agent -> time to transfer results
                       of one job needed by another between two agents

[1]. http://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time
"""

from functools import partial
from collections import namedtuple
from .util import reverse_dict
from itertools import chain

Event = namedtuple('Event', 'job start end')

def wbar(ni, agents, compcost):
    """ Average computation cost """
    return sum(compcost(ni, agent) for agent in agents) / len(agents)

def cbar(ni, nj, agents, commcost):
    """ Average communication cost """
    n = len(agents)
    if n == 1:
        return 0
    npairs = n * (n-1)
    return 1. * sum(commcost(ni, nj, a1, a2) for a1 in agents for a2 in agents
                                        if a1 != a2) / npairs

def ranku(ni, agents, succ,  compcost, commcost):
    """ Rank of job

    This code is designed to mirror the wikipedia entry.
    Please see that for details

    [1]. http://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time
    """
    rank = partial(ranku, compcost=compcost, commcost=commcost,
                           succ=succ, agents=agents)
    w = partial(wbar, compcost=compcost, agents=agents)
    c = partial(cbar, agents=agents, commcost=commcost)

    if ni in succ and succ[ni]:
        return w(ni) + max(c(ni, nj) + rank(nj) for nj in succ[ni])
    else:
        return w(ni)

def endtime(job, events):
    """ Endtime of job in list of events """
    for e in events:
        if e.job == job:
            return e.end

def find_first_gap(agent_orders, desired_start_time, duration):
    """Find the first gap in an agent's list of jobs

    The gap must be after `desired_start_time` and of length at least
    `duration`.
    """

    # No jobs: can fit it in whenever the job is ready to run
    if (agent_orders is None) or (len(agent_orders)) == 0:
        return desired_start_time;

    # Try to fit it in between each pair of Events, but first prepend a
    # dummy Event which ends at time 0 to check for gaps before any real
    # Event starts.
    a = chain([Event(None,None,0)], agent_orders[:-1])
    for e1, e2 in zip(a, agent_orders):
        earliest_start = max(desired_start_time, e1.end)
        if e2.start - earliest_start > duration:
            return earliest_start

    # No gaps found: put it at the end, or whenever the task is ready
    return max(agent_orders[-1].end, desired_start_time)

def start_time(job, orders, jobson, prec, commcost, compcost, agent):
    """ Earliest time that job can be executed on agent """

    duration = compcost(job, agent)

    if job in prec:
        comm_ready = max([endtime(p, orders[jobson[p]])
                       + commcost(p, job, agent, jobson[p]) for p in prec[job]])
    else:
        comm_ready = 0

    return find_first_gap(orders[agent], comm_ready, duration)

def allocate(job, orders, jobson, prec, compcost, commcost):
    """ Allocate job to the machine with earliest finish time

    Operates in place
    """
    st = partial(start_time, job, orders, jobson, prec, commcost, compcost)
    ft = lambda machine: st(machine) + compcost(job, machine)

    agent = min(orders.keys(), key=ft)
    start = st(agent)
    end = ft(agent)

    orders[agent].append(Event(job, start, end))
    orders[agent] = sorted(orders[agent], key=lambda e: e.start)
    # Might be better to use a different data structure to keep each
    # agent's orders sorted at a lower cost.

    jobson[job] = agent

def makespan(orders):
    """ Finish time of last job """
    return max(v[-1].end for v in orders.values() if v)

def schedule(succ, agents, compcost, commcost):
    """ Schedule computation dag onto worker agents

    inputs:

    succ - DAG of tasks {a: (b, c)} where b, and c follow a
    agents - set of agents that can perform work
    compcost - function :: job, agent -> runtime
    commcost - function :: j1, j2, a1, a2 -> communication time
    """
    rank = partial(ranku, agents=agents, succ=succ,
                          compcost=compcost, commcost=commcost)
    prec = reverse_dict(succ)

    jobs = set(succ.keys()) | set(x for xx in succ.values() for x in xx)
    jobs = sorted(jobs, key=rank)

    orders = {agent: [] for agent in agents}
    jobson = dict()
    for job in reversed(jobs):
        allocate(job, orders, jobson, prec, compcost, commcost)

    return orders, jobson


def recvs(job, jobson, prec, recv):
    """ Collect all necessary recvs for job """
    if job not in prec:
        return []
    return [recv(jobson[p], jobson[job], p, job) for p in prec[job]
                if jobson[p] != jobson[job]]

def sends(job, jobson, succ, send):
    """ Collect all necessary sends for job """
    if job not in succ:
        return []
    return [send(jobson[job], jobson[s], job, s) for s in succ[job]
                if jobson[s] != jobson[job]]

eps = 1e-9
def insert_recvs(order, jobson, prec, recv):
    if not order:
        return order

    thisagent = jobson[order[0].job]

    receives = partial(recvs, jobson=jobson, prec=prec, recv=recv)
    recv_events = {e.job: [Event(r, e.start, e.start)
                                    for r in receives(e.job)]
                          for e in order}

    for job, revents in recv_events.items():
        i = [e.job for e in order].index(job)
        order = order[:i] + revents + order[i:]

    jobson.update({e.job: thisagent for es in recv_events.values() for e in es})

    return order

def insert_sends(order, jobson, succ, send):
    if not order:
        return order

    thisagent = jobson[order[0].job]

    sends2 = partial(sends, jobson=jobson, succ=succ, send=send)
    send_events = {e.job: [Event(s, e.end, e.end)
                                    for s in sends2(e.job)]
                          for e in order}

    for job, sevents in send_events.items():
        i = [e.job for e in order].index(job)
        order = order[:i+1] + sevents + order[i+1:]

    jobson.update({e.job: thisagent for es in send_events.values() for e in es})

    return order

def insert_sendrecvs(orders, jobson, succ, send, recv):
    """ Insert send an recv events into the orders at approprate places """
    prec = reverse_dict(succ)
    jobson = jobson.copy()
    neworders = dict()
    for agent, order in orders.items():
        order = insert_sends(order, jobson, succ, send)
        order = insert_recvs(order, jobson, prec, recv)
        neworders[agent] = order
    return neworders, jobson

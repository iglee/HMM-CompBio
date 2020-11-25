import numpy as np

def update_emit_proba(path, seq):
    
    # emit proba from sequence at state 0
    seq0 = seq[path == 0]
    state0 = np.array([(seq0 == 0).sum(), (seq0 == 1).sum(), (seq0 == 2).sum(), (seq0 == 3).sum()])/seq0.shape[0]
    
    # emit proba from sequence at state 1
    seq1 = seq[path == 1]
    state1 = np.array([(seq1 == 0).sum(), (seq1 == 1).sum(), (seq1 == 2).sum(), (seq1 == 3).sum()])/seq1.shape[0]
    
    return np.array([state0, state1])



def update_trans_proba(path, n_intervals):
    total0 = (path == 0).sum()
    total1 = (path == 1).sum()
    
    
    return np.array([[(total0 - 2*n_intervals)/total0, (2*n_intervals)/total0],\
                     [(2*n_intervals)/total1, (total1 - 2*n_intervals)/total1]])


def vt_train(h):
    """
    input h = HMM object
    """
    new_emit_proba = update_emit_proba(h.path, h.seq_idx)
    new_trans_proba = update_trans_proba(h.path,len(h.intervals))
    h.update_proba(new_emit_proba, new_trans_proba)
    h.viterbi()
    h.backtrace()
    return h
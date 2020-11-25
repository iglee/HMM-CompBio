import numpy as np

def update_emit_proba(path, seq):
    return None

def update_trans_proba(path, n_intervals):
    return None


def bw_train(h):
    """
    input h = HMM object
    """
    new_emit_proba = update_emit_proba(h.path, h.seq_idx)
    new_trans_proba = update_trans_proba(h.path,len(h.intervals))
    h.update_proba(new_emit_proba, new_trans_proba)
    h.viterbi()
    h.backtrace()
    return h
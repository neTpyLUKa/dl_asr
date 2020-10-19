import numpy as np


def _levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    distance = np.zeros((2, n + 1), dtype=np.int32)

    for j in range(0, n + 1):
        distance[0][j] = j

    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def char_errors(answer, target):
    answer = ' '.join(filter(None, answer.split(' ')))
    target = ' '.join(filter(None, target.split(' ')))

    edit_distance = _levenshtein_distance(answer, target)
    return float(edit_distance), len(target)


def calculate_cer(answers, targets):
    cers = []
    for i in range(len(answers)):
        edit_distance, target_len = char_errors(answers[i], targets[i])

        cers.append(float(edit_distance) / target_len)
    return cers


def word_errors(answer, target):
    answer_words = answer.split()
    target_words = target.split()

    edit_distance = _levenshtein_distance(answer_words, target_words)
    return float(edit_distance), len(target_words)


def calculate_wer(answers, targets):
    wers = []
    for i in range(len(answers)):
        edit_distance, target_len = word_errors(answers[i], targets[i])

        wer = float(edit_distance) / target_len
        wers.append(wer)
    return wers

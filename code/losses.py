import tensorflow as tf


def transe_score(hs, rs, ts):
    distance = hs + rs - ts
    score = tf.reduce_sum(tf.square(distance), axis=1)
    # score = tf.reduce_sum(tf.abs(distance), axis=1)
    return -score


def mde_score(hs, rs, ts, gamma=12):
    a = hs[0] + rs[0] - ts[0]
    b = hs[1] + ts[1] - rs[1]
    c = ts[2] + rs[2] - hs[2]
    d = hs[3] - rs[3] * ts[3]

    e = hs[4] + rs[4] - ts[4]
    f = hs[5] + ts[5] - rs[5]
    g = ts[6] + rs[6] - hs[6]
    i = hs[7] - rs[7] * ts[7]

    score_a = (tf.norm(a, ord=2, axis=1) + tf.norm(e, ord=2, axis=1)) / 2.0
    score_b = (tf.norm(b, ord=2, axis=1) + tf.norm(f, ord=2, axis=1)) / 2.0
    score_c = (tf.norm(c, ord=2, axis=1) + tf.norm(g, ord=2, axis=1)) / 2.0
    score_d = (tf.norm(d, ord=2, axis=1) + tf.norm(i, ord=2, axis=1)) / 2.0
    # score_a = tf.reduce_sum((a + e) / 2.0, axis=1)
    # score_b = tf.reduce_sum((b + f) / 2.0, axis=1)
    # score_c = tf.reduce_sum((c + g) / 2.0, axis=1)
    # score_d = tf.reduce_sum((d + i) / 2.0, axis=1)
    score = (1.5 * score_a + 3.0 * score_b + 1.5 * score_c + 3.0 * score_d) / 9.0
    # score = gamma - score
    return -score


def logistic_loss(phs, prs, pts, nhs, nrs, nts, mode='transe', pws=None, nws=None):
    if mode == 'transe':
        pos_score = transe_score(phs[0], prs[0], pts[0])
        neg_score = transe_score(nhs[0], nrs[0], nts[0])
    elif mode == 'mde':
        pos_score = mde_score(phs, prs, pts)
        neg_score = mde_score(nhs, nrs, nts)
    pos_score = tf.log(1 + tf.exp(-pos_score))
    neg_score = tf.log(1 + tf.exp(neg_score))
    if None not in (pws, nws):
        pos_score = tf.multiply(pos_score, pws)
        neg_score = tf.multiply(neg_score, nws)
    pos_loss = tf.reduce_sum(pos_score)
    neg_loss = tf.reduce_sum(neg_score)
    loss = tf.add(pos_loss, neg_loss)
    return loss


def margin_loss(phs, prs, pts, nhs, nrs, nts, margin, mode='transe'):
    if mode == 'transe':
        pos_score = transe_score(phs[0], prs[0], pts[0])
        neg_score = transe_score(nhs[0], nrs[0], nts[0])
    elif mode == 'mde':
        pos_score = mde_score(phs, prs, pts)
        neg_score = mde_score(nhs, nrs, nts)
    loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) - pos_score + neg_score))
    return loss


def limited_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, mode='transe', balance=1.0):
    if mode == 'transe':
        pos_score = transe_score(phs[0], prs[0], pts[0])
        neg_score = transe_score(nhs[0], nrs[0], nts[0])
    elif mode == 'mde':
        pos_score = mde_score(phs, prs, pts)
        neg_score = mde_score(nhs, nrs, nts)
    pos_loss = tf.reduce_sum(tf.nn.relu(-pos_score - tf.constant(pos_margin)))
    neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) + neg_score))
    loss = tf.add(pos_loss, balance * neg_loss)
    return loss


def positive_logistic_loss(phs, prs, pts, mode='transe', pws=None):
    if mode == 'transe':
        pos_score = transe_score(phs[0], prs[0], pts[0])
    elif mode == 'mde':
        pos_score = mde_score(phs, prs, pts)
    pos_score = tf.log(1 + tf.exp(-pos_score))
    if pws is not None:
        pos_score = tf.multiply(pos_score, pws)
    pos_loss = tf.reduce_sum(pos_score)
    return pos_loss


def positive_margin_loss(phs, prs, pts, margin, mode='transe', pws=None):
    if mode == 'transe':
        pos_score = transe_score(phs[0], prs[0], pts[0])
    elif mode == 'mde':
        pos_score = mde_score(phs, prs, pts)
    if pws is not None:
        pos_score = tf.multiply(pos_score, pws)
    pos_loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) - pos_score))
    return pos_loss


def positive_limited_loss(phs, prs, pts, margin, mode='transe', pws=None):
    if mode == 'transe':
        pos_score = transe_score(phs[0], prs[0], pts[0])
    elif mode == 'mde':
        pos_score = mde_score(phs, prs, pts)
    if pws is not None:
        pos_score = tf.multiply(pos_score, pws)
    pos_loss = tf.reduce_sum(tf.nn.relu(-pos_score - tf.constant(margin)))
    return pos_loss


def space_mapping_loss(view_embeds, shared_embeds, mapping, eye, orthogonal_weight, norm_w=0.0001):
    mapped_ents2 = tf.matmul(view_embeds, mapping)
    mapped_ents2 = tf.nn.l2_normalize(mapped_ents2)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.square(shared_embeds - mapped_ents2), 1))
    norm_loss = tf.reduce_sum(tf.reduce_sum(tf.square(mapping), 1))
    return map_loss + orthogonal_weight * orthogonal_loss(mapping, eye) + norm_w * norm_loss


def orthogonal_loss(mapping, eye):
    loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return loss


def alignment_loss(ents1, ents2):
    distance = ents1 - ents2
    loss = tf.reduce_sum(tf.reduce_sum(tf.square(distance), axis=1))
    return loss

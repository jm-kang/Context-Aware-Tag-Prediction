import argparse
import json
import numpy as np

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def recall_at_k(r, k, g):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.sum(r) / g


def p_at_k(results, test, tagset, k):
    p_at_k_list = []
    for post in test:
        pid = post['id']
        generated_tags = results[pid]
        gold_tags = post['post']['post_tags']
        gold_tags = [tag for tag in gold_tags if tag in tagset]
        r = [1 if tag in gold_tags else 0 for tag in generated_tags]
        p_at_k_list.append(precision_at_k(r, k))
    return (sum(p_at_k_list) / len(p_at_k_list)), p_at_k_list


def r_at_k(results, test, tagset, k):
    r_at_k_list = []
    for post in test:
        pid = post['id']
        generated_tags = results[pid]
        gold_tags = post['post']['post_tags']
        gold_tags = [tag for tag in gold_tags if tag in tagset]
        if len(gold_tags) == 0:
            r_at_k_list.append(0.0)
            continue
        r = [1 if tag in gold_tags else 0 for tag in generated_tags]
        g = len(gold_tags)
        r_at_k_list.append(recall_at_k(r, k, g))
    return (sum(r_at_k_list) / len(r_at_k_list)), r_at_k_list


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--result_file", default=None, type=str, required=True,
                        help="Name of file that contains the results of prediction.")
    parser.add_argument("--eval_data_file", default='dataset/test.json', type=str, required=True,
                        help="evaluation data file.")
    parser.add_argument("--tag_list", default='dataset/tagset.txt', type=str, required=True,
                        help="tag set to consider.")

    args = parser.parse_args()

    with open(args.result_file) as f:
        results = json.load(f)
    with open(args.eval_data_file) as f:
        test = json.load(f)
    with open(args.tag_list) as f:
        tagset = f.readlines()
        tagset = [tag.replace('\n', '') for tag in tagset]

    k_list = [1, 3, 5, 10]
    for k in k_list:
        print('P@'+str(k)+':', p_at_k(results, test, tagset, k)[0].round(4))
    for k in k_list:
        print('R@'+str(k)+':', r_at_k(results, test, tagset, k)[0].round(4))

if __name__ == "__main__":
    main()

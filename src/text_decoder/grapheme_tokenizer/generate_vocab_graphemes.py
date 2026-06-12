

"""
Choose a vocab of fixed size that maximizes the coverage
"""

import json


def get_coverage(tel: dict, san: dict, vocab: dict | list, vocab_size: int):
    # Get total tokens count
    tel_count = sum([tel[graph] for graph in tel])
    san_count = sum([san[graph] for graph in san])

    # Sort the vocab by count/relative frequency
    if isinstance(vocab, dict):
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        vocab = [tup[0] for tup in vocab]

    selected_vocab = vocab[:vocab_size]

    # Calculate coverage
    token_count = 0
    for graph in selected_vocab:
        token_count += tel.get(graph, 0)
        token_count += san.get(graph, 0)

    coverage = token_count/(tel_count + san_count)

    print(f"The coverage with the selected vocab size of {vocab_size} is {coverage*100:.2f}%")
    return coverage*100



def get_top_common(tel, san, top):
    tel = sorted(tel.items(), key=lambda x: x[1], reverse=True)[:top]
    san = sorted(san.items(), key=lambda x: x[1], reverse=True)[:top]

    tel_top = [tup[0] for tup in tel]
    san_top = [tup[0] for tup in san]

    common = len(set(tel_top).intersection(set(san_top)))
    common = (common/top)*100

    print(f"The common graphemes percentage in top {top} is: {common:.2f}%")

    return common




if __name__ == '__main__':

    VOCAB = 2043      # Leave some space for special tokens like EOS, PAD etc

    with open("token_dist/english_grapheme_dist.json", 'r') as f:
        eng_dist = json.load(f)


    with open("token_dist/sanskrit_grapheme_dist.json", 'r') as f:
        san_dist = json.load(f)


    with open("token_dist/telugu_grapheme_dist.json", 'r') as f:
        tel_dist = json.load(f)


    # remove some graphemes for fair calculations
    graphemes_to_remove = [' ', '.', '\n']
    for graph in graphemes_to_remove:
        tel_dist.pop(graph)
        san_dist.pop(graph)

    eng_tokens = sum([eng_dist[graph] for graph in eng_dist])
    san_tokens = sum([san_dist[graph] for graph in san_dist])
    tel_tokens = sum([tel_dist[graph] for graph in tel_dist])


    print(f"English: Unique graphemes: {len(eng_dist):<5}\t\t Total graphemes: {eng_tokens}")
    print(f"Sanskrit: Unique graphemes: {len(san_dist):<5}\t\t Total graphemes: {san_tokens}")
    print(f"Telugu: Unique graphemes: {len(tel_dist):<5}\t\t Total graphemes: {tel_tokens}\n\n")


    # Weigh the frequencies of Telugu and Sanskrit distributions by the content length
    # Scheme 1: Total count / total freq
    unique_graphemes = set(list(san_dist.keys())).union(set(list(tel_dist.keys())))

    rel_graph_count = {}
    for graph in unique_graphemes:
        rel_count = (san_dist.get(graph, 0) + tel_dist.get(graph, 0))/(san_tokens + tel_tokens)
        rel_graph_count[graph] = rel_count

    # Sort the rel_graph_count
    rel_graph_count = dict(sorted(rel_graph_count.items(), key=lambda x: x[1], reverse=True))

    # get_coverage(tel_dist, san_dist, rel_graph_count, 2048-229)
    # get_top_common(tel_dist, san_dist, 2048-229)

    # print('='*50)

    # Scheme 2: Weighted sum of the counts
    tel_wei, san_wei = 0.8, 0.2
    wei_graph_count = {}

    for graph in unique_graphemes:
        wei_count = (tel_dist.get(graph, 0)/tel_tokens) * tel_wei + (san_dist.get(graph, 0)/san_tokens) * san_wei
        wei_graph_count[graph] = wei_count

    wei_graph_count = dict(sorted(wei_graph_count.items(), key=lambda x: x[1], reverse=True))

    # get_coverage(tel_dist, san_dist, wei_graph_count, 2048-229)
    # get_top_common(tel_dist, san_dist, 2048-229)


    # # Now form the vocab
    # ========== 1) Include English Characters ==========

    # Instead of depending on the calculated English graphemes,
    # enumerate them and add all the possible graphemes (to not miss any grapheme)
    printable_english_chars = "\t\n\r !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    finalized_vocab = []
    for char in printable_english_chars:
        finalized_vocab.append(char)

    # ========== 2) Telugu Unicodes (for fallback) ==========
    # Include fall back Telugu base unicodes
    lst = list(range(ord('\u0C00'), ord('\u0C7F') + 1)) + [ord('\u0964'), ord('\u0965'), ord('\u1CDA')]

    for code in lst:
        finalized_vocab.append(chr(code))

    # ========== 3) Add the combined top graphemes from Telugu and Sanskrit ==========
    # Add the top graphemes to the remaining vocab
    rem_vocab = VOCAB - len(finalized_vocab)
    grapheme_vocab = [tup[0] for tup in rel_graph_count.items()]

    i = 0
    while len(finalized_vocab) < VOCAB:
        if grapheme_vocab[i] not in finalized_vocab:
            finalized_vocab.append(grapheme_vocab[i])
        i += 1

    with open("grapheme_list.txt", 'w') as f:
        for grapheme in finalized_vocab:
            f.write(repr(grapheme)+'\n')

    print(finalized_vocab)

    get_coverage(tel_dist, san_dist, finalized_vocab, VOCAB)
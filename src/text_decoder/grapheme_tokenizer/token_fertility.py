"""
Calculate the character token fertility of the tokenizer.

This is useful to estimate the number of tokens given the number of characters.
Used for estimating the mix languages in the training dataset based on the estimate of tokens
"""

import json


def calculate_fertility(tokenizer_vocab, dist):
    num_tokens, num_codes = 0, 0

    for grapheme in dist:
        count = dist[grapheme]

        # Update num_tokens
        if grapheme in tokenizer_vocab:
            num_tokens += count
        else:
            num_tokens += count * len(grapheme)

        # Update num_codes
        num_codes += count * len(grapheme)

    fertility = num_codes / num_tokens

    return fertility


if __name__ == '__main__':

    with open("telugu-vocab.json", 'r') as f:
        vocab = json.load(f)

    # Estimate the token fertility for sanskrit
    with open("token_dist/sanskrit_grapheme_dist.json", 'r') as f:
        san_dist = json.load(f)

    with open("token_dist/telugu_grapheme_dist.json", 'r') as f:
        tel_dist = json.load(f)


    san_fertility = calculate_fertility(vocab, san_dist)
    tel_fertility = calculate_fertility(vocab, tel_dist)

    print(f"Sanskrit fertility: {san_fertility:.4f}")
    print(f"Telugu   fertility: {tel_fertility:.4f}")
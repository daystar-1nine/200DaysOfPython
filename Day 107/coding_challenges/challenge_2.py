def post_pad_sequences(sequences, max_len, pad_value=0):
    """
    Pads sequences with pad_value at the end until max_len.
    """
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_value] * (max_len - len(seq)))
    return padded

if __name__ == "__main__":
    seqs = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]
    print(post_pad_sequences(seqs, 4))

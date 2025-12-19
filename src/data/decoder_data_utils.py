import numpy as np

def tokenize_fusionsmi(text):
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == "[":
            j = text.find("]", i)
            if j == -1:
                tokens.append(text[i])
                i += 1
            else:
                tokens.append(text[i:j+1])
                i = j + 1
        elif text[i] == "{":
            j = text.find("}", i)
            if j == -1:
                tokens.append(text[i])
                i += 1
            else:
                tokens.append("{")
                inner = text[i+1:j].strip()
                if inner:
                    tokens.extend(inner.split())
                tokens.append("}")
                i = j + 1
        else:
            if text[i] != " ":
                tokens.append(text[i])
            i += 1
    return tokens

def tokenize_pad_fusionsmi(seq_list, vocab):
    tokenized_seqs = [tokenize_fusionsmi(seq) for seq in seq_list]
    token2id = vocab["token2id"]
    pad_id = token2id.get("<PAD>", 0)

    token_ids = [[token2id.get(tok, token2id["<UNK>"]) for tok in tokens] for tokens in tokenized_seqs]

    max_len = max(len(ids) for ids in token_ids)
    padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in token_ids]

    mask = [[1]*len(ids) + [0]*(max_len - len(ids)) for ids in token_ids]

    return np.array(padded_ids), np.array(mask)

aa_1to3 = {
    'A':'Ala', 'C':'Cys', 'D':'Asp', 'E':'Glu', 'F':'Phe',
    'G':'Gly', 'H':'His', 'I':'Ile', 'K':'Lys', 'L':'Leu',
    'M':'Met', 'N':'Asn', 'P':'Pro', 'Q':'Gln', 'R':'Arg',
    'S':'Ser', 'T':'Thr', 'V':'Val', 'W':'Trp', 'Y':'Tyr'
}

def tokenize_pad_prot_sequence(seq_list, vocab, max_len=None):
    token2id = vocab["token2id"]
    pad_id = token2id.get("<PAD>", 0)
    
    tokenized_seqs = []
    for seq in seq_list:
        tokens = []
        for aa in seq:
            three_aa = aa_1to3.get(aa, "<UNK>")
            tokens.append(three_aa)
        tokens = tokens[:max_len]
        tokenized_seqs.append(tokens)

    token_ids = [[token2id.get(tok, token2id["<UNK>"]) for tok in tokens] for tokens in tokenized_seqs]
    
    cur_max_len = min(max(len(ids) for ids in token_ids), max_len)
    padded_ids = [ids + [pad_id] * (cur_max_len - len(ids)) for ids in token_ids]

    mask = [[1]*len(ids) + [0]*(cur_max_len - len(ids)) for ids in token_ids]
    
    return np.array(padded_ids), np.array(mask)
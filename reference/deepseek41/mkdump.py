"""mkdump.py TEXT N OUT_DIR LAYERS...: a dump directory `nextsycl llm check` runs on without a reference's values - the
first N tokens of TEXT (DeepSeek-V4.1's tokenizer, after the BOS) in tokens.txt and an index of the taps to save
(NS_CHECK_SAVE) for LAYERS (l_in, pre_mix, attn_out, ffn_out, l_out, comp at sources), their files empty: the engine's
tensors are saved for a teacher-forced reference run (ds41ref.py --teacher)."""
import os
import sys

import tokenizers

text, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
layers = [int(x) for x in sys.argv[4:]]
tok = tokenizers.Tokenizer.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json'))
ids = [0] + tok.encode(open(text).read(), add_special_tokens=False).ids
ids = ids[:n]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'tokens.txt'), 'w').write(' '.join(map(str, ids)) + '\n')
D = 5120
with open(os.path.join(out, 'index.tsv'), 'w') as f:
    for l in layers:
        for name, dims in [(f'l_in-{l}', [D, 4, n]), (f'pre_mix-{l}', [4, n]), (f'attn_out-{l}', [D, n]), (f'ffn_out-{l}', [D, n]), (f'l_out-{l}', [D, 4, n])]:
            d = dims + [1] * (4 - len(dims))
            f.write('\t'.join([name, 'f32'] + [str(x) for x in d]) + '\n')
            open(os.path.join(out, name + '.f32'), 'wb').close()
print(len(ids), 'tokens')

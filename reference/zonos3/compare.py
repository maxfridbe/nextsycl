"""Compare `nextsycl audio check DIR --model zonos3-mini` outputs (nc_*.npy) with the reference's dumps in DIR (ref.sh).

    python3 reference/zonos3/compare.py DIR

prompt ids (ns_ids.npy vs ids.npy), the backbone's logits (ns_logits.npy vs ggml/logits.npy from zonos3-cli --validate),
teacher-forced logits (ns_forced.npy vs logits.npy from zonos3-cli --tts --dump-logits), the decoder's PCM (ns_pcm.npy vs
zac_pcm.npy from zac-cli) and the speaker embedding (ns_embedding.npy vs spk_emb.npy from spk-encoder-cli). Comparison
only: nothing here runs in nextsycl.
"""
import numpy as np, sys, os
d = sys.argv[1]
def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / (np.linalg.norm(b.astype(np.float64)) + 1e-30))
j = lambda f: os.path.join(d, f)
if os.path.exists(j('ns_ids.npy')):
    a, b = np.load(j('ns_ids.npy')), np.load(j('ids.npy'))
    print('prompt ids identical:', a.shape == b.shape and bool((a == b).all()), a.shape, b.shape)
if os.path.exists(j('ns_logits.npy')) and os.path.exists(j('ggml/logits.npy')):
    a, b = np.load(j('ns_logits.npy')), np.load(j('ggml/logits.npy')).reshape(np.load(j('ns_logits.npy')).shape)
    rows = [cos(a[r], b[r]) for r in range(len(a))]
    am = (a.argmax(-1) == b.argmax(-1)).mean()
    print(f'lm logits: cos min {min(rows):.6f} mean {np.mean(rows):.6f}, rel {rel(a,b):.2e}, argmax agree {am*100:.1f}%')
if os.path.exists(j('ns_forced.npy')) and os.path.exists(j('logits.npy')):
    a, b = np.load(j('ns_forced.npy')), np.load(j('logits.npy'))
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    rows = [cos(a[r], b[r]) for r in range(n)]
    am = (a.argmax(-1) == b.argmax(-1))
    print(f'forced logits ({n} steps): cos min {min(rows):.6f} mean {np.mean(rows):.6f}, rel {rel(a,b):.2e}, argmax agree {am.mean()*100:.1f}% ({am.sum()}/{am.size})')
    # the sampled codes: does each drawn code rank near the top of ours?
    rws = np.load(j('out.npy'))[3:3 + n].astype(int)
    ranks = []
    for s in range(n):
        for c in range(10):
            t = rws[s][c]
            if t < 4096:
                ranks.append(int((a[s][c] > a[s][c][t]).sum()))
    ranks = np.array(ranks)
    print(f'  the reference take\'s codes in our logits: top-1 {np.mean(ranks==0)*100:.1f}%, top-80 {np.mean(ranks<80)*100:.1f}%')
if os.path.exists(j('ns_pcm.npy')) and os.path.exists(j('zac_pcm.npy')):
    a, b = np.load(j('ns_pcm.npy')), np.load(j('zac_pcm.npy'))
    n = min(len(a), len(b))
    print(f'decoder pcm: {len(a)} vs {len(b)} samples, cos {cos(a[:n], b[:n]):.6f}, rel {rel(a[:n], b[:n]):.2e}, max abs diff {float(np.abs(a[:n]-b[:n]).max()):.2e}')
if os.path.exists(j('ns_embedding.npy')) and os.path.exists(j('spk_emb.npy')):
    a, b = np.load(j('ns_embedding.npy')), np.load(j('spk_emb.npy'))
    print(f'speaker embedding: cos {cos(a,b):.6f}, rel {rel(a,b):.2e}, norms {np.linalg.norm(a):.4f} / {np.linalg.norm(b):.4f}')

"""Lip sync of a clip, SyncNet's way (Chung & Zisserman, joonson/syncnet_python at 907c0b57, its weights syncnet_v2.model
and the S3FD face detector sfd_face.pth): faces found in every frame (S3FD), kept as tracks by position (one a speaker),
each track's face crop (the pipeline's crop: the box squared, 40% padding, 224 px) against the sound's MFCCs - the
distance between their SyncNet features at audio offsets -15..15 frames; the offset is the best one, the confidence the
median distance less the best (SyncNet's AV offset and confidence; a good sync is offset 0-1 and confidence above ~5).

The frames and the sound come cut beforehand (ffmpeg, the build image): the clip at SyncNet's 25 fps as PNGs, its sound
16 kHz mono. Writes one JSON line a clip: each track's offset, confidence, frames, and its face boxes (for mouth crops).

    python -I lipsync.py --syncnet syncnet_python --clip NAME --frames DIR --wav audio.wav [--tracks 2 --spans 0:A-B,1:C-D]

With --spans each speaker of a two-shot is scored only over the seconds it speaks (its own Whisper word times): over the
whole clip the listener's silent half pulls the confidence down.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import python_speech_features
import torch
from scipy import signal
from scipy.io import wavfile


def calc_pdist(feat1, feat2, vshift=15):
    """syncnet_python's: for each video window, the distance to the audio windows shifted -vshift..vshift"""
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))
    return dists


def main():
    ap = argparse.ArgumentParser()
    for k in ('syncnet', 'clip', 'frames', 'wav'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--tracks', type=int, default=1, help='speakers to follow (faces kept by position, left to right)')
    ap.add_argument('--spans', default='', help='each track scored only while it speaks: "0:0.2-3.1,1:3.0-6.4" (seconds)')
    a = ap.parse_args()
    sys.path.insert(0, a.syncnet)
    os.chdir(a.syncnet)                                      # S3FD reads its weights from ./detectors/s3fd/weights
    torch.set_num_threads(8)
    from SyncNetModel import S
    from detectors import S3FD
    det = S3FD(device='cpu')
    net = S(num_layers_in_fc_layers=1024)
    state = torch.load(os.path.join(a.syncnet, 'data/syncnet_v2.model'), map_location='cpu')
    own = net.state_dict()
    for k, v in state.items():
        own[k].copy_(v)
    net.eval()

    paths = sorted(glob.glob(os.path.join(os.path.abspath(a.frames), '*.png')))
    frames = [cv2.imread(p) for p in paths]
    # faces a frame: S3FD at the pipeline's scale, the best few
    boxes = []
    for f in frames:
        b = det.detect_faces(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), conf_th=0.9, scales=[0.25])
        b = sorted([list(map(float, x[:4])) for x in b], key=lambda x: -(x[2] - x[0]) * (x[3] - x[1]))[:a.tracks]
        boxes.append(sorted(b, key=lambda x: x[0]))        # left to right
    # a track a speaker: the k-th face from the left in frames that have all of them; a missing face takes the last box
    tracks = []
    for t in range(a.tracks):
        seq, last = [], None
        for b in boxes:
            if len(b) == a.tracks:
                last = b[t]
            seq.append(last)
        if any(x is None for x in seq):
            first = next((x for x in seq if x is not None), None)
            if first is None:
                continue
            seq = [x if x is not None else first for x in seq]
        tracks.append(seq)

    sr, audio = wavfile.read(a.wav)
    mfcc = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(audio, sr))])
    cc = torch.from_numpy(np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0).astype(float)).float()
    out = {'clip': a.clip, 'frames': len(frames), 'tracks': []}
    for ti, seq in enumerate(tracks):
        # the pipeline's crop: box centre, size = max side / 2 (each median-filtered over 13 frames), 40% padding, 224 px
        bs = 0.4
        crops = []
        sz = signal.medfilt([max(x2 - x1, y2 - y1) / 2 for x1, y1, x2, y2 in seq], kernel_size=13)
        xs = signal.medfilt([(x1 + x2) / 2 for x1, y1, x2, y2 in seq], kernel_size=13)
        ys = signal.medfilt([(y1 + y2) / 2 for x1, y1, x2, y2 in seq], kernel_size=13)
        for f, s, cx, cy in zip(frames, sz, xs, ys):
            bsi = int(s * (1 + 2 * bs))
            pad = np.pad(f, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my, mx = cy + bsi, cx + bsi
            face = pad[int(my - s):int(my + s * (1 + 2 * bs)), int(mx - s * (1 + bs)):int(mx + s * (1 + bs))]
            crops.append(cv2.resize(face, (224, 224)))
        im = np.stack(crops, axis=3)
        im = np.expand_dims(im, axis=0).transpose((0, 3, 4, 1, 2))
        imtv = torch.from_numpy(im.astype(float)).float()
        lastframe = min(len(crops), int(len(audio) / sr * 25)) - 5
        first = 0
        for sp in filter(None, a.spans.split(',')):
            k, rng = sp.split(':')
            if int(k) == ti:
                t0, t1 = map(float, rng.split('-'))
                first, lastframe = max(0, int(t0 * 25)), min(lastframe, int(t1 * 25) - 2)
        with torch.no_grad():
            im_feat = torch.cat([net.forward_lip(imtv[:, :, i:i + 5, :, :]) for i in range(first, lastframe)], 0)
            cc_feat = torch.cat([net.forward_aud(cc[:, :, :, i * 4:i * 4 + 20]) for i in range(first, lastframe)], 0)
        dists = calc_pdist(im_feat, cc_feat, vshift=15)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        minval, minidx = torch.min(mdist, 0)
        offset = 15 - int(minidx)
        conf = float(torch.median(mdist) - minval)
        # per window: the confidence at the best offset (where in the clip the sync holds)
        fdist = torch.stack([d[int(minidx)] for d in dists])
        fconf = torch.median(mdist) - fdist
        out['tracks'].append({'track': ti, 'windows': [first, lastframe], 'offset': offset, 'confidence': round(conf, 3), 'min_dist': round(float(minval), 3),
                              'frame_conf_mean': round(float(fconf.mean()), 3), 'boxes': [[round(v, 1) for v in b] for b in seq]})
    print(json.dumps(out))


if __name__ == '__main__':
    main()

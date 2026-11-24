/** The player: the film's live HLS playlist while it is made (hls.js from jsDelivr where the browser has no HLS of its
 *  own; Safari's native), its mp4 once it is done. Attached once a source, again only when the source changes. */

const HLS_JS = "https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js";

interface HlsLike {
  loadSource(u: string): void;
  attachMedia(v: HTMLVideoElement): void;
  destroy(): void;
}
interface HlsCtor {
  new (cfg?: Record<string, unknown>): HlsLike;
  isSupported(): boolean;
}

let loading: Promise<HlsCtor | null> | null = null;
function hlsJs(): Promise<HlsCtor | null> {
  loading ??= new Promise((ok) => {
    const s = document.createElement("script");
    s.src = HLS_JS;
    s.onload = () => ok((window as unknown as { Hls?: HlsCtor }).Hls ?? null);
    s.onerror = () => ok(null);
    document.head.appendChild(s);
  });
  return loading;
}

const attached = new WeakMap<HTMLVideoElement, { src: string; hls: HlsLike | null }>();

export function attach(v: HTMLVideoElement, src: string | null, live: boolean): void {
  if (!src || attached.get(v)?.src === src) return;
  attached.get(v)?.hls?.destroy();
  attached.set(v, { src, hls: null });
  if (!live || v.canPlayType("application/vnd.apple.mpegurl")) {
    v.src = src;
    void v.play().catch(() => {});
    return;
  }
  void hlsJs().then((Hls) => {
    if (attached.get(v)?.src !== src) return;
    if (!Hls || !Hls.isSupported()) {
      v.src = src;
      return;
    }
    const h = new Hls({ liveDurationInfinity: true });
    h.loadSource(src);
    h.attachMedia(v);
    attached.set(v, { src, hls: h });
    void v.play().catch(() => {});
  });
}

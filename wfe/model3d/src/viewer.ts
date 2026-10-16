/** A GLB viewer on WebGL2, no library: every triangle primitive of the scene (its node transforms applied), its
 *  base colour texture or vertex colours, lit by a key light and the sky; a skeleton drawn over it as lines; a morph
 *  animation (one target a frame, as the server writes it) played. Orbit with a drag, zoom with the wheel or a
 *  pinch, pan with the right button or two fingers. */

type Mat4 = Float32Array;

const ident = (): Mat4 => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += (a[k * 4 + r] ?? 0) * (b[c * 4 + k] ?? 0);
      o[c * 4 + r] = s;
    }
  return o;
}

function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect; o[5] = f; o[10] = (far + near) / (near - far); o[11] = -1; o[14] = (2 * far * near) / (near - far);
  return o;
}

function lookAt(eye: number[], at: number[], up: number[]): Mat4 {
  const sub = (a: number[], b: number[]) => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
  const norm = (a: number[]) => { const l = Math.hypot(a[0]!, a[1]!, a[2]!) || 1; return [a[0]! / l, a[1]! / l, a[2]! / l]; };
  const cross = (a: number[], b: number[]) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
  const dot = (a: number[], b: number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0]!, y[0]!, z[0]!, 0, x[1]!, y[1]!, z[1]!, 0, x[2]!, y[2]!, z[2]!, 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

/** A node's local matrix: `matrix`, else T * R * S */
function local(n: Record<string, unknown>): Mat4 {
  const m = n["matrix"] as number[] | undefined;
  if (m && m.length === 16) return new Float32Array(m);
  const t = (n["translation"] as number[] | undefined) ?? [0, 0, 0];
  const q = (n["rotation"] as number[] | undefined) ?? [0, 0, 0, 1];
  const s = (n["scale"] as number[] | undefined) ?? [1, 1, 1];
  const [x, y, z, w] = [q[0]!, q[1]!, q[2]!, q[3]!];
  const r = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
             2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
             2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)];
  const o = ident();
  for (let c = 0; c < 3; c++) for (let rr = 0; rr < 3; rr++) o[c * 4 + rr] = r[c * 3 + rr]! * s[c]!;
  o[12] = t[0]!; o[13] = t[1]!; o[14] = t[2]!;
  return o;
}

const apply = (m: Mat4, p: ArrayLike<number>, i: number, w: number): [number, number, number] => [
  m[0]! * p[i]! + m[4]! * p[i + 1]! + m[8]! * p[i + 2]! + m[12]! * w,
  m[1]! * p[i]! + m[5]! * p[i + 1]! + m[9]! * p[i + 2]! + m[13]! * w,
  m[2]! * p[i]! + m[6]! * p[i + 1]! + m[10]! * p[i + 2]! + m[14]! * w,
];

interface Part {
  pos: Float32Array; nrm: Float32Array; uv: Float32Array | null; col: Float32Array | null; idx: Uint32Array;
  image: ImageBitmap | null; factor: number[];
  /** morph frames: each target's position offsets, in world space */
  targets: Float32Array[];
}

export interface Loaded {
  triangles: number; vertices: number; textured: boolean; joints: number; frames: number; fps: number;
}

/** A GLB's parts (world space), skeleton lines and animation clock */
async function parse(buf: ArrayBuffer): Promise<{ parts: Part[]; bones: Float32Array; fps: number; frames: number }> {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("not a GLB file");
  const jl = dv.getUint32(12, true);
  const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl))) as Record<string, any>;
  const at = 20 + jl;
  const bin = at + 8 <= buf.byteLength ? new Uint8Array(buf, at + 8, dv.getUint32(at, true)) : new Uint8Array(0);
  const view = (i: number) => {
    const v = doc.bufferViews[i];
    return { off: bin.byteOffset + (v.byteOffset ?? 0), len: v.byteLength as number, stride: (v.byteStride ?? 0) as number };
  };
  const SIZE: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
  /** an accessor as floats (normalized integers to [0, 1]) */
  const floats = (i: number): { a: Float32Array; n: number } => {
    const acc = doc.accessors[i];
    const n = SIZE[acc.type as string] ?? 1;
    const count = acc.count as number;
    const out = new Float32Array(count * n);
    if (acc.bufferView === undefined) return { a: out, n };
    const v = view(acc.bufferView);
    const ct = acc.componentType as number;
    const sz = ct === 5126 || ct === 5125 ? 4 : ct === 5123 || ct === 5122 ? 2 : 1;
    const stride = v.stride || sz * n;
    const d = new DataView(buf, v.off + (acc.byteOffset ?? 0));
    for (let e = 0; e < count; e++)
      for (let c = 0; c < n; c++) {
        const p = e * stride + c * sz;
        let x: number;
        switch (ct) {
          case 5126: x = d.getFloat32(p, true); break;
          case 5125: x = d.getUint32(p, true); break;
          case 5123: x = d.getUint16(p, true); if (acc.normalized) x /= 65535; break;
          case 5122: x = d.getInt16(p, true); if (acc.normalized) x = Math.max(x / 32767, -1); break;
          case 5121: x = d.getUint8(p); if (acc.normalized) x /= 255; break;
          default: x = d.getInt8(p); if (acc.normalized) x = Math.max(x / 127, -1);
        }
        out[e * n + c] = x;
      }
    return { a: out, n };
  };
  const images = new Map<number, ImageBitmap | null>();
  const image = async (tex: Record<string, any> | undefined): Promise<ImageBitmap | null> => {
    if (!tex || tex["index"] === undefined) return null;
    const src = doc.textures?.[tex["index"]]?.source;
    if (src === undefined) return null;
    if (images.has(src)) return images.get(src) ?? null;
    const im = doc.images[src];
    let bm: ImageBitmap | null = null;
    if (im.bufferView !== undefined) {
      const v = view(im.bufferView);
      bm = await createImageBitmap(new Blob([new Uint8Array(buf, v.off, v.len)], { type: im.mimeType ?? "image/png" }), { imageOrientation: "none" });
    }
    images.set(src, bm);
    return bm;
  };
  const parts: Part[] = [];
  const bones: number[] = [];
  const scene = doc.scenes?.[doc.scene ?? 0]?.nodes ?? doc.nodes.map((_: unknown, i: number) => i);
  const stack: [number, Mat4][] = (scene as number[]).map((r) => [r, ident()]);
  while (stack.length) {
    const [ni, parent] = stack.pop()!;
    const node = doc.nodes[ni];
    const world = mul(parent, local(node));
    for (const k of (node.children ?? []) as number[]) stack.push([k, world]);
    if (node.mesh === undefined) continue;
    const mesh = doc.meshes[node.mesh];
    for (const p of mesh.primitives as Record<string, any>[]) {
      if ((p["mode"] ?? 4) !== 4 || p["attributes"].POSITION === undefined) continue;
      const P = floats(p["attributes"].POSITION).a;
      const nv = P.length / 3;
      const pos = new Float32Array(nv * 3);
      for (let i = 0; i < nv; i++) pos.set(apply(world, P, i * 3, 1), i * 3);
      let idx: Uint32Array;
      if (p["indices"] !== undefined) idx = Uint32Array.from(floats(p["indices"]).a);
      else idx = Uint32Array.from({ length: nv }, (_, i) => i);
      let nrm: Float32Array;
      if (p["attributes"].NORMAL !== undefined) {
        const N = floats(p["attributes"].NORMAL).a;
        nrm = new Float32Array(nv * 3);
        for (let i = 0; i < nv; i++) nrm.set(apply(world, N, i * 3, 0), i * 3);
      } else {
        nrm = new Float32Array(nv * 3);
        for (let t = 0; t + 2 < idx.length; t += 3) {
          const [a, b, c] = [idx[t]! * 3, idx[t + 1]! * 3, idx[t + 2]! * 3];
          const e1 = [pos[b]! - pos[a]!, pos[b + 1]! - pos[a + 1]!, pos[b + 2]! - pos[a + 2]!];
          const e2 = [pos[c]! - pos[a]!, pos[c + 1]! - pos[a + 1]!, pos[c + 2]! - pos[a + 2]!];
          const f = [e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!];
          for (const v of [a, b, c]) for (let k = 0; k < 3; k++) nrm[v + k]! += f[k]!;
        }
      }
      const uv = p["attributes"].TEXCOORD_0 !== undefined ? floats(p["attributes"].TEXCOORD_0).a : null;
      let col: Float32Array | null = null;
      if (p["attributes"].COLOR_0 !== undefined) {
        const c = floats(p["attributes"].COLOR_0);
        col = new Float32Array(nv * 4);
        for (let i = 0; i < nv; i++) for (let k = 0; k < 4; k++) col[i * 4 + k] = k < c.n ? c.a[i * c.n + k]! : 1;
      }
      const mat = p["material"] !== undefined ? doc.materials[p["material"]] : undefined;
      const pbr = mat?.pbrMetallicRoughness ?? {};
      const targets: Float32Array[] = [];
      for (const t of (p["targets"] ?? []) as Record<string, number>[]) {
        if (t["POSITION"] === undefined) continue;
        const D = floats(t["POSITION"]).a;
        const w = new Float32Array(nv * 3);
        for (let i = 0; i < nv; i++) w.set(apply(world, D, i * 3, 0), i * 3);
        targets.push(w);
      }
      parts.push({ pos, nrm, uv, col, idx, image: await image(pbr.baseColorTexture), factor: pbr.baseColorFactor ?? [1, 1, 1, 1], targets });
    }
  }
  // the skeleton: each joint's head from its inverse bind matrix's translation (inverted), a line to its parent
  const skin = doc.skins?.[0];
  if (skin && skin.inverseBindMatrices !== undefined) {
    const ibm = floats(skin.inverseBindMatrices).a;
    const joints = skin.joints as number[];
    const head = joints.map((_, k) => {
      const m = ibm.subarray(k * 16, k * 16 + 16);
      // affine inverse of the translation part, as the server writes them (pure translations)
      return [-m[12]!, -m[13]!, -m[14]!];
    });
    joints.forEach((n, k) => {
      for (const c of (doc.nodes[n].children ?? []) as number[]) {
        const j = joints.indexOf(c);
        if (j >= 0) bones.push(...head[k]!, ...head[j]!);
      }
    });
  }
  const anim = doc.animations?.[0]?.samplers?.[0];
  let fps = 24;
  if (anim) {
    const t = floats(anim.input).a;
    if (t.length > 1 && t[1]! > 0) fps = 1 / t[1]!;
  }
  const frames = Math.max(0, ...parts.map((p) => p.targets.length));
  return { parts, bones: new Float32Array(bones), fps, frames };
}

const VS = `#version 300 es
in vec3 aPos; in vec3 aNrm; in vec2 aUv; in vec4 aCol;
uniform mat4 uVP; out vec3 vNrm; out vec2 vUv; out vec4 vCol; out vec3 vPos;
void main(){ vNrm=aNrm; vUv=aUv; vCol=aCol; vPos=aPos; gl_Position=uVP*vec4(aPos,1.0); }`;
const FS = `#version 300 es
precision highp float;
in vec3 vNrm; in vec2 vUv; in vec4 vCol; in vec3 vPos; out vec4 o;
uniform sampler2D uTex; uniform int uHasTex; uniform vec4 uFactor; uniform vec3 uEye; uniform int uFlat;
void main(){
  vec3 n = uFlat==1 ? normalize(cross(dFdx(vPos), dFdy(vPos))) : normalize(vNrm);
  vec3 v = normalize(uEye - vPos); if (dot(n, v) < 0.0) n = -n;
  vec4 base = uFactor * vCol; if (uHasTex==1) base *= texture(uTex, vUv);
  vec3 l = normalize(vec3(0.5, 0.9, 0.6));
  float sky = 0.5 + 0.5 * n.y;
  vec3 h = normalize(l + v);
  vec3 c = base.rgb * (0.22 + 0.33 * sky + 0.62 * max(dot(n, l), 0.0)) + vec3(0.08) * pow(max(dot(n, h), 0.0), 40.0);
  o = vec4(pow(c, vec3(1.0)), 1.0);
}`;
const LVS = `#version 300 es
in vec3 aPos; uniform mat4 uVP; void main(){ gl_Position = uVP*vec4(aPos,1.0); gl_PointSize = 6.0; }`;
const LFS = `#version 300 es
precision highp float; uniform vec4 uColor; out vec4 o; void main(){ o = uColor; }`;

interface GpuPart { vao: WebGLVertexArrayObject; posBuf: WebGLBuffer; count: number; tex: WebGLTexture | null; factor: number[]; wire: WebGLBuffer; wireCount: number; src: Part }

export class Viewer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private lprog: WebGLProgram;
  private parts: GpuPart[] = [];
  private bones: { buf: WebGLBuffer; count: number } | null = null;
  private center = [0, 0, 0];
  private radius = 1;
  private yaw = 0.6;
  private pitch = 0.35;
  private dist = 3;
  private pan = [0, 0, 0];
  private raf = 0;
  private frames = 0;
  private fps = 24;
  private t0 = performance.now();
  private observer: ResizeObserver;
  wire = false;
  flat = false;
  skeleton = true;
  playing = true;
  url = "";

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.prog = this.program(VS, FS);
    this.lprog = this.program(LVS, LFS);
    this.controls();
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    this.loop();
  }

  private program(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
    return p;
  }

  /** Shows the GLB at `url` (its stats when it loads) */
  async load(url: string): Promise<Loaded> {
    this.url = url;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    const { parts, bones, fps, frames } = await parse(await r.arrayBuffer());
    if (this.url !== url) throw new Error("superseded");
    this.clear();
    const gl = this.gl;
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    let tris = 0, verts = 0, textured = false;
    for (const p of parts) {
      for (let i = 0; i < p.pos.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k]!, p.pos[i + k]!); hi[k] = Math.max(hi[k]!, p.pos[i + k]!); }
      tris += p.idx.length / 3;
      verts += p.pos.length / 3;
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const attr = (name: string, data: Float32Array | null, n: number, fallback: number[]) => {
        const loc = gl.getAttribLocation(this.prog, name);
        if (loc < 0) return null;
        if (!data) {
          gl.disableVertexAttribArray(loc);
          gl.vertexAttrib4f(loc, fallback[0]!, fallback[1]!, fallback[2]!, fallback[3]!);
          return null;
        }
        const b = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
        return b;
      };
      const posBuf = attr("aPos", p.pos, 3, [0, 0, 0, 1])!;
      attr("aNrm", p.nrm, 3, [0, 0, 1, 0]);
      attr("aUv", p.uv, 2, [0, 0, 0, 0]);
      attr("aCol", p.col, 4, [0.78, 0.8, 0.84, 1]);
      const ib = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, p.idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      let tex: WebGLTexture | null = null;
      if (p.image && p.uv) {
        tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, p.image);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        textured = true;
      }
      // the wireframe: every triangle's edges as lines
      const edges = new Uint32Array(p.idx.length * 2);
      for (let t = 0; t + 2 < p.idx.length; t += 3)
        for (let k = 0; k < 3; k++) { edges[(t + k) * 2] = p.idx[t + k]!; edges[(t + k) * 2 + 1] = p.idx[t + (k + 1) % 3]!; }
      const wire = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wire);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edges, gl.STATIC_DRAW);
      this.parts.push({ vao, posBuf, count: p.idx.length, tex, factor: p.factor, wire, wireCount: edges.length, src: p });
    }
    if (bones.length) {
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, bones, gl.STATIC_DRAW);
      this.bones = { buf: b, count: bones.length / 3 };
    }
    if (!Number.isFinite(lo[0]!)) { lo = [-1, -1, -1]; hi = [1, 1, 1]; }
    this.center = [0, 1, 2].map((k) => (lo[k]! + hi[k]!) / 2);
    this.radius = Math.max(Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!) / 2, 1e-3);
    this.dist = this.radius * 2.6;
    this.pan = [0, 0, 0];
    this.frames = frames;
    this.fps = fps;
    this.t0 = performance.now();
    this.draw();
    return { triangles: tris, vertices: verts, textured, joints: this.bones ? this.bones.count / 2 : 0, frames, fps };
  }

  private clear(): void {
    const gl = this.gl;
    for (const p of this.parts) {
      gl.deleteVertexArray(p.vao);
      if (p.tex) gl.deleteTexture(p.tex);
      gl.deleteBuffer(p.wire);
    }
    this.parts = [];
    if (this.bones) gl.deleteBuffer(this.bones.buf);
    this.bones = null;
  }

  private controls(): void {
    const c = this.canvas;
    const pointers = new Map<number, { x: number; y: number }>();
    let pinch = 0;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); pointers.set(e.pointerId, { x: e.clientX, y: e.clientY }); });
    const up = (e: PointerEvent) => { pointers.delete(e.pointerId); pinch = 0; };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("pointermove", (e) => {
      const last = pointers.get(e.pointerId);
      if (!last) return;
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a!.x - b!.x, a!.y - b!.y);
        if (pinch) this.dist = Math.min(Math.max(this.dist * (pinch / d), this.radius * 0.3), this.radius * 20);
        pinch = d;
        this.panBy(dx / 2, dy / 2);
      } else if (e.buttons & 2 || e.shiftKey) {
        this.panBy(dx, dy);
      } else {
        this.yaw -= dx * 0.008;
        this.pitch = Math.min(Math.max(this.pitch + dy * 0.008, -1.5), 1.5);
      }
      this.draw();
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.dist = Math.min(Math.max(this.dist * Math.exp(e.deltaY * 0.0012), this.radius * 0.3), this.radius * 20);
      this.draw();
    }, { passive: false });
    c.addEventListener("dblclick", () => { this.yaw = 0.6; this.pitch = 0.35; this.dist = this.radius * 2.6; this.pan = [0, 0, 0]; this.draw(); });
  }

  private panBy(dx: number, dy: number): void {
    const s = this.dist * 0.0015;
    const right = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
    this.pan = [this.pan[0]! - right[0]! * dx * s, this.pan[1]! + dy * s, this.pan[2]! - right[2]! * dx * s];
  }

  /** A redraw every frame only while an animation plays */
  private loop(): void {
    this.raf = requestAnimationFrame(() => this.loop());
    if (this.frames > 1 && this.playing) this.draw();
  }

  /** The morph frame now: the positions blended between the two frames around the clock */
  private morph(): void {
    if (this.frames < 2) return;
    const gl = this.gl;
    const t = ((performance.now() - this.t0) / 1000) * this.fps;
    const f = t % this.frames;
    const a = Math.floor(f), b = (a + 1) % this.frames, w = f - a;
    for (const p of this.parts) {
      const A = p.src.targets[a], B = p.src.targets[b];
      if (!A || !B) continue;
      const out = new Float32Array(p.src.pos.length);
      for (let i = 0; i < out.length; i++) out[i] = p.src.pos[i]! + A[i]! * (1 - w) + B[i]! * w;
      gl.bindBuffer(gl.ARRAY_BUFFER, p.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, out, gl.DYNAMIC_DRAW);
    }
  }

  draw(): void {
    const gl = this.gl, c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.043, 0.051, 0.067, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    if (this.playing) this.morph();
    const target = [this.center[0]! + this.pan[0]!, this.center[1]! + this.pan[1]!, this.center[2]! + this.pan[2]!];
    const eye = [target[0]! + this.dist * Math.cos(this.pitch) * Math.sin(this.yaw), target[1]! + this.dist * Math.sin(this.pitch),
                 target[2]! + this.dist * Math.cos(this.pitch) * Math.cos(this.yaw)];
    const vp = mul(perspective(0.75, w / h, this.radius * 0.01, this.radius * 60), lookAt(eye, target, [0, 1, 0]));
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, "uVP"), false, vp);
    gl.uniform3fv(gl.getUniformLocation(this.prog, "uEye"), eye);
    gl.uniform1i(gl.getUniformLocation(this.prog, "uFlat"), this.flat ? 1 : 0);
    for (const p of this.parts) {
      gl.bindVertexArray(p.vao);
      gl.uniform4fv(gl.getUniformLocation(this.prog, "uFactor"), p.factor);
      gl.uniform1i(gl.getUniformLocation(this.prog, "uHasTex"), p.tex ? 1 : 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, p.tex);
      gl.enable(gl.POLYGON_OFFSET_FILL);
      gl.polygonOffset(1, 1);
      gl.drawElements(gl.TRIANGLES, p.count, gl.UNSIGNED_INT, 0);
      gl.disable(gl.POLYGON_OFFSET_FILL);
    }
    gl.bindVertexArray(null);
    gl.useProgram(this.lprog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.lprog, "uVP"), false, vp);
    const lpos = gl.getAttribLocation(this.lprog, "aPos");
    if (this.wire) {
      gl.uniform4fv(gl.getUniformLocation(this.lprog, "uColor"), [0.36, 0.55, 0.95, 0.9]);
      for (const p of this.parts) {
        gl.bindBuffer(gl.ARRAY_BUFFER, p.posBuf);
        gl.enableVertexAttribArray(lpos);
        gl.vertexAttribPointer(lpos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, p.wire);
        gl.drawElements(gl.LINES, p.wireCount, gl.UNSIGNED_INT, 0);
      }
    }
    if (this.skeleton && this.bones) {
      gl.disable(gl.DEPTH_TEST);
      gl.uniform4fv(gl.getUniformLocation(this.lprog, "uColor"), [0.98, 0.62, 0.1, 1]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bones.buf);
      gl.enableVertexAttribArray(lpos);
      gl.vertexAttribPointer(lpos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, this.bones.count);
      gl.drawArrays(gl.POINTS, 0, this.bones.count);
    }
    gl.disableVertexAttribArray(lpos);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.observer.disconnect();
    this.clear();
  }
}

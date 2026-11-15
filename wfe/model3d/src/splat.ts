/** 3D Gaussian splats on WebGL2, no library: a 3DGS PLY read (binary little-endian floats: x y z, f_dc_0..2, opacity as
 *  its logit, scale_0..2 as logs, rot_0..3 a (w, x, y, z) quaternion), each splat drawn as a screen-space quad its
 *  projected covariance shapes (EWA, the 0.3 px low-pass of the reference rasterizer), blended back to front after a
 *  16-bit counting sort by depth, redone when the view turns. The PLY is in 3DGS's frame (y down: the object's up is -y);
 *  it is turned 180 degrees about x on reading, into the viewer's y up. */

export interface SplatData {
  count: number;
  /** per splat: x y z, r g b a (0..1), the 3-D covariance's xx xy xz yy yz zz - 13 floats */
  rows: Float32Array;
  lo: number[];
  hi: number[];
}

const SH_C0 = 0.28209479177387814;
const STRIDE = 13;

/** A 3DGS PLY's splats (the transparent ones, opacity under 1/255, left out) */
export function parsePly(buf: ArrayBuffer): SplatData {
  const bytes = new Uint8Array(buf);
  const marker = "end_header\n";
  let end = -1;
  for (let i = 0; i < Math.min(bytes.length, 65536) - marker.length; i++) {
    let ok = true;
    for (let k = 0; k < marker.length && ok; k++) ok = bytes[i + k] === marker.charCodeAt(k);
    if (ok) { end = i + marker.length; break; }
  }
  if (end < 0) throw new Error("not a PLY file (no end_header)");
  const head = new TextDecoder().decode(bytes.subarray(0, end));
  if (!head.startsWith("ply") || !head.includes("binary_little_endian")) throw new Error("a binary little-endian PLY is shown here");
  let n = 0;
  const props: string[] = [];
  let inVertex = false;
  for (const line of head.split("\n")) {
    const w = line.trim().split(/\s+/);
    if (w[0] === "element") { inVertex = w[1] === "vertex"; if (inVertex) n = parseInt(w[2] ?? "0", 10); }
    else if (w[0] === "property" && inVertex) {
      if (w[1] !== "float" && w[1] !== "float32") throw new Error(`${line}: float properties only`);
      props.push(w[2] ?? "");
    }
  }
  const stride = props.length;
  const at = (p: string) => {
    const i = props.indexOf(p);
    if (i < 0) throw new Error(`the PLY has no ${p}`);
    return i;
  };
  const ix = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"].map(at);
  if (buf.byteLength < end + n * stride * 4) throw new Error(`${n} splats do not fit the file`);
  // the float data may sit at an offset that is not a multiple of 4: copy it out
  const f = new Float32Array(bytes.slice(end, end + n * stride * 4).buffer);
  const rows = new Float32Array(n * STRIDE);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let m = 0;
  for (let i = 0; i < n; i++) {
    const g = (k: number) => f[i * stride + ix[k]!]!;
    const a = 1 / (1 + Math.exp(-g(6)));
    if (!(a > 1 / 255)) continue;
    const o = m * STRIDE;
    // 3DGS's y down to the viewer's y up: (x, y, z) -> (x, -y, -z), the covariance's xy and xz with it
    const x = g(0), y = -g(1), z = -g(2);
    rows[o] = x; rows[o + 1] = y; rows[o + 2] = z;
    for (let c = 0; c < 3; c++) rows[o + 3 + c] = Math.min(Math.max(0.5 + SH_C0 * g(3 + c), 0), 1);
    rows[o + 6] = a;
    // covariance = (R S)(R S)^T
    const s = [Math.exp(g(7)), Math.exp(g(8)), Math.exp(g(9))];
    let qw = g(10), qx = g(11), qy = g(12), qz = g(13);
    const ql = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= ql; qx /= ql; qy /= ql; qz /= ql;
    const R = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
               2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
               2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)];
    const M = R.map((v, k) => v * s[k % 3]!);
    const cov = (r: number, c: number) => M[r * 3]! * M[c * 3]! + M[r * 3 + 1]! * M[c * 3 + 1]! + M[r * 3 + 2]! * M[c * 3 + 2]!;
    rows[o + 7] = cov(0, 0); rows[o + 8] = -cov(0, 1); rows[o + 9] = -cov(0, 2);
    rows[o + 10] = cov(1, 1); rows[o + 11] = cov(1, 2); rows[o + 12] = cov(2, 2);
    lo[0] = Math.min(lo[0]!, x); lo[1] = Math.min(lo[1]!, y); lo[2] = Math.min(lo[2]!, z);
    hi[0] = Math.max(hi[0]!, x); hi[1] = Math.max(hi[1]!, y); hi[2] = Math.max(hi[2]!, z);
    m++;
  }
  return { count: m, rows: rows.subarray(0, m * STRIDE), lo, hi };
}

const VS = `#version 300 es
precision highp float;
in vec2 aCorner; in vec3 aCenter; in vec4 aColor; in vec3 aCovA; in vec3 aCovB;
uniform mat4 uView; uniform mat4 uProj; uniform vec2 uViewport; uniform float uFocal;
out vec4 vColor; out vec2 vPos;
void main() {
  vec4 cam = uView * vec4(aCenter, 1.0);
  vec4 clip = uProj * cam;
  float zc = -cam.z;
  float lim = 1.3 * clip.w;
  if (zc < 0.01 || abs(clip.x) > lim || abs(clip.y) > lim) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  mat3 S = mat3(aCovA.x, aCovA.y, aCovA.z, aCovA.y, aCovB.x, aCovB.y, aCovA.z, aCovB.y, aCovB.z);
  mat3 W = mat3(uView);
  mat3 C = W * S * transpose(W);
  // the projection's Jacobian at the centre, in pixels (rows: u, v; columns: the camera's x, y, z)
  vec3 ju = vec3(uFocal / zc, 0.0, uFocal * cam.x / (zc * zc));
  vec3 jv = vec3(0.0, uFocal / zc, uFocal * cam.y / (zc * zc));
  float a = dot(ju, C * ju) + 0.3, b = dot(ju, C * jv), c = dot(jv, C * jv) + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  vec2 e1 = abs(b) > 1e-12 ? normalize(vec2(b, l1 - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 e2 = vec2(-e1.y, e1.x);
  vec2 off = aCorner.x * 3.0 * sqrt(l1) * e1 + aCorner.y * 3.0 * sqrt(l2) * e2;
  vColor = aColor;
  vPos = 3.0 * aCorner;
  gl_Position = vec4(clip.xy / clip.w + off * 2.0 / uViewport, 0.0, 1.0);
}`;
const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 o;
void main() {
  float q = dot(vPos, vPos);
  if (q > 9.0) discard;
  float al = min(0.99, vColor.a * exp(-0.5 * q));
  if (al < 1.0 / 255.0) discard;
  o = vec4(vColor.rgb * al, al);
}`;

/** The splats on the card: their rows in draw order (far to near), re-sorted when the view direction moves */
export class SplatLayer {
  private prog: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private buf: WebGLBuffer;
  private quad: WebGLBuffer;
  private sorted: Float32Array;
  private lastDir = [0, 0, 0];
  private lastEye = [0, 0, 0];

  constructor(private gl: WebGL2RenderingContext, program: (vs: string, fs: string) => WebGLProgram, readonly data: SplatData) {
    this.prog = program(VS, FS);
    this.sorted = new Float32Array(data.rows.length);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    this.quad = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
    const corner = gl.getAttribLocation(this.prog, "aCorner");
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    this.buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, this.sorted, gl.DYNAMIC_DRAW);
    const inst = (name: string, n: number, off: number) => {
      const loc = gl.getAttribLocation(this.prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, STRIDE * 4, off * 4);
      gl.vertexAttribDivisor(loc, 1);
    };
    inst("aCenter", 3, 0);
    inst("aColor", 4, 3);
    inst("aCovA", 3, 7);
    inst("aCovB", 3, 10);
    gl.bindVertexArray(null);
  }

  /** The rows far to near along `dir` from `eye` (a counting sort of 16-bit depths) */
  private sort(eye: number[], dir: number[]): void {
    const { count, rows } = this.data;
    const depth = new Float32Array(count);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE;
      const d = (rows[o]! - eye[0]!) * dir[0]! + (rows[o + 1]! - eye[1]!) * dir[1]! + (rows[o + 2]! - eye[2]!) * dir[2]!;
      depth[i] = d;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    const scale = hi > lo ? 65535 / (hi - lo) : 0;
    const counts = new Uint32Array(65536);
    const key = new Uint16Array(count);
    for (let i = 0; i < count; i++) {
      // far first: the largest depth gets the smallest key
      const k = 65535 - Math.min(65535, Math.max(0, Math.floor((depth[i]! - lo) * scale)));
      key[i] = k;
      counts[k]!++;
    }
    let sum = 0;
    for (let k = 0; k < 65536; k++) { const c = counts[k]!; counts[k] = sum; sum += c; }
    for (let i = 0; i < count; i++) {
      const at = counts[key[i]!]!++;
      this.sorted.set(rows.subarray(i * STRIDE, (i + 1) * STRIDE), at * STRIDE);
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.sorted);
    this.lastDir = dir;
    this.lastEye = eye;
  }

  draw(view: Float32Array, proj: Float32Array, eye: number[], target: number[], w: number, h: number, fovy: number): void {
    const gl = this.gl;
    const d = [target[0]! - eye[0]!, target[1]! - eye[1]!, target[2]! - eye[2]!];
    const l = Math.hypot(d[0]!, d[1]!, d[2]!) || 1;
    const dir = d.map((v) => v / l);
    const turned = dir[0]! * this.lastDir[0]! + dir[1]! * this.lastDir[1]! + dir[2]! * this.lastDir[2]! < 0.9995;
    const moved = Math.hypot(eye[0]! - this.lastEye[0]!, eye[1]! - this.lastEye[1]!, eye[2]! - this.lastEye[2]!) > l * 0.02;
    if (turned || moved) this.sort(eye, dir);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, "uView"), false, view);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, "uProj"), false, proj);
    gl.uniform2f(gl.getUniformLocation(this.prog, "uViewport"), w, h);
    gl.uniform1f(gl.getUniformLocation(this.prog, "uFocal"), 0.5 * h / Math.tan(fovy / 2));
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.data.count);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.buf);
    gl.deleteBuffer(this.quad);
    gl.deleteProgram(this.prog);
  }
}

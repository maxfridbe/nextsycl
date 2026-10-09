"""ComfyUI's samplers and schedulers on a flow (CONST) model, for crates/diffusion's tests.

Runs ComfyUI's own code (comfy/k_diffusion/sampling.py, deis.py, extra_samplers/uni_pc.py and the scheduler functions of
comfy/samplers.py, loaded from a ComfyUI checkout with the heavy imports stubbed) on a small analytic denoiser and a
fixed noise sequence, and writes what they return to expected.txt next to this file. The Rust tests in
crates/diffusion/src/samplers.rs and schedule.rs read that file and compare.

    python comfy_ref.py /path/to/ComfyUI > expected.txt     (CPU only: torch, numpy, scipy, tqdm)

expected.txt lines (whitespace separated, floats printed with repr):
    noise <v...>                    the noise sequence: each noise_sampler call takes the next len(x) values
    x <v...>                        the starting latent
    sigmas <set> <v...>             a sigma schedule the samplers run on (model sampling shift 3 for "shift3", else 1)
    sampler <set> <name> <v...>     ComfyUI's result (after KSAMPLER's inverse_noise_scaling) for that sampler and set
    schedule <name> <steps> <shift> <v...>   ComfyUI's sigmas for a scheduler on ModelSamplingDiscreteFlow(shift)
"""
import ast
import inspect
import math
import sys
import types

import numpy
import scipy.integrate
import scipy.stats
import torch

COMFY = sys.argv[1] if len(sys.argv) > 1 else "/var/mnt/2TBSSD/minimaxh3/repo/vendor/ComfyUI"
N = 16  # latent size


def load(path, names=None, env=None):
    """exec a ComfyUI source file without its imports (env supplies what it uses); names: only these top-level defs"""
    tree = ast.parse(open(path).read())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if names is not None and not (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names):
            continue
        body.append(node)
    tree.body = body
    ns = dict(env or {})
    exec(compile(tree, path, "exec"), ns)
    return types.SimpleNamespace(**ns)


# --- ComfyUI's flow model sampling (comfy/model_sampling.py: CONST + ModelSamplingDiscreteFlow), float32 as there ---
def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class CONST:
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        s = getattr(self, "noise_scale", 1.0)
        return sigma * (s * noise) + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


class FlowSampling(CONST):
    def __init__(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.noise_scale = 1.0
        self.shift = shift
        self.multiplier = multiplier
        self.sigmas = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


comfy = types.SimpleNamespace(model_sampling=types.SimpleNamespace(CONST=CONST))
utils = types.SimpleNamespace(append_dims=lambda x, n: x[(...,) + (None,) * (n - x.ndim)])
trange = lambda n, disable=None: range(n)
env = dict(torch=torch, math=math, np=numpy, numpy=numpy, nn=torch.nn, partial=__import__("functools").partial,
           integrate=scipy.integrate, scipy=scipy, comfy=comfy, utils=utils, trange=torch and trange,
           tqdm=lambda *a, **k: __import__("contextlib").nullcontext(types.SimpleNamespace(update=lambda *a: None)),
           logging=__import__("logging"))
env["deis"] = load(f"{COMFY}/comfy/k_diffusion/deis.py", env=env)
env["sa_solver"] = None
env["torchsde"] = None  # every SDE sampler here gets our noise_sampler, never a BrownianTree
kd = load(f"{COMFY}/comfy/k_diffusion/sampling.py", env=env)
unipc = load(f"{COMFY}/comfy/extra_samplers/uni_pc.py", env=env)
sched = load(f"{COMFY}/comfy/samplers.py", names={"simple_scheduler", "ddim_scheduler", "normal_scheduler",
             "beta_scheduler", "linear_quadratic_schedule", "kl_optimal_scheduler"}, env=env)


def denoiser(x, sigma):
    """the analytic stand-in for a model's x0 prediction (crates/diffusion's tests use the same)"""
    i = torch.arange(x.shape[-1], dtype=x.dtype)
    s = sigma.reshape(-1, 1)
    return torch.tanh(0.9 * x + 0.03 * i) * (1 - s) + 0.1 * s


class Model:
    """what the k-diffusion samplers reach through: model(x, sigma) and the model_sampling"""
    def __init__(self, ms):
        self.inner_model = types.SimpleNamespace(
            inner_model=types.SimpleNamespace(model_sampling=ms),
            model_patcher=types.SimpleNamespace(get_model_object=lambda name: ms))

    def __call__(self, x, sigma, **kwargs):
        # the model sees float32 x and sigma, as the Rust samplers hand them over (and as ComfyUI's float32 latents
        # and sigmas are): near sigma 1, 1 - sigma in float32 is coarse enough to matter at 1e-4
        return denoiser(x.float().double(), sigma.float().double())


SAMPLERS = ["euler", "euler_ancestral", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2",
            "dpm_2_ancestral", "lms", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde",
            "dpmpp_2m_sde_heun", "dpmpp_3m_sde", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "res_multistep",
            "res_multistep_ancestral", "gradient_estimation", "er_sde", "seeds_2", "seeds_3", "ddim", "uni_pc",
            "uni_pc_bh2"]


def sampler_function(name):
    if name == "uni_pc":
        return unipc.sample_unipc
    if name == "uni_pc_bh2":
        return unipc.sample_unipc_bh2
    if name == "ddim":
        return kd.sample_euler  # ComfyUI: ksampler("euler", inpaint_options={"random": True})
    return getattr(kd, f"sample_{name}")


def fmt(v):
    return " ".join(repr(float(a)) for a in v)


def main():
    rng = numpy.random.RandomState(20261009)
    noise_seq = rng.standard_normal(N * 200)
    x0 = rng.standard_normal(N)
    print("noise", fmt(noise_seq))
    print("x", fmt(x0))

    ms = FlowSampling(shift=1.0)
    shift3 = time_snr_shift(3.0, torch.linspace(1, 0, 11, dtype=torch.float32))
    sets = {
        "shift3": shift3,
        "karras6": kd.get_sigmas_karras(6, float(ms.sigma_min), float(ms.sigma_max)).float(),
        "img2img": shift3[3:].clone(),
    }
    for set_name, sig in sets.items():
        # the model sampling's shift: only offset_first_sigma_for_snr's percent_to_sigma sees it
        ms = FlowSampling(shift=3.0 if set_name == "shift3" else 1.0)
        sig = sig.float()
        print("sigmas", set_name, fmt(sig))
        for name in SAMPLERS:
            pos = [0]

            def noise_sampler(sigma, sigma_next):
                v = noise_seq[pos[0]:pos[0] + N]
                pos[0] += N
                return torch.tensor(v, dtype=torch.float64).reshape(1, N)

            fn = sampler_function(name)
            sigmas = sig.double().clone()
            x = torch.tensor(x0, dtype=torch.float64).reshape(1, N)
            kwargs = {"extra_args": {}, "disable": True}
            if "noise_sampler" in inspect.signature(fn).parameters:
                kwargs["noise_sampler"] = noise_sampler
            torch.manual_seed(0)
            # uni_pc builds its coefficient tensors with torch.tensor(list): float64 here so they meet our float64 x
            # (ComfyUI runs float32 latents and gets float32 ones)
            torch.set_default_dtype(torch.float64 if name.startswith("uni_pc") else torch.float32)
            out = fn(Model(ms), x, sigmas, **kwargs)
            torch.set_default_dtype(torch.float32)
            out = ms.inverse_noise_scaling(sigmas[-1], out)  # KSAMPLER.sample (sigmas as the sampler left them)
            print("sampler", set_name, name, fmt(out.reshape(-1)))

    for shift in [1.0, 3.0, 6.0]:
        ms = FlowSampling(shift=shift)
        for steps in [1, 4, 7, 20, 30]:
            for name, f in [
                ("simple", lambda: sched.simple_scheduler(ms, steps)),
                ("sgm_uniform", lambda: sched.normal_scheduler(ms, steps, sgm=True)),
                ("karras", lambda: kd.get_sigmas_karras(n=steps, sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max))),
                ("exponential", lambda: kd.get_sigmas_exponential(n=steps, sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max))),
                ("ddim_uniform", lambda: sched.ddim_scheduler(ms, steps)),
                ("beta", lambda: sched.beta_scheduler(ms, steps)),
                ("normal", lambda: sched.normal_scheduler(ms, steps)),
                ("linear_quadratic", lambda: sched.linear_quadratic_schedule(ms, steps)),
                ("kl_optimal", lambda: sched.kl_optimal_scheduler(n=steps, sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max))),
            ]:
                if name == "kl_optimal" and steps == 1:
                    continue  # ComfyUI divides by steps - 1: NaN
                print("schedule", name, steps, shift, fmt(f().float()))


main()

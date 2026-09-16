"""Compare the actual C++ library against PyTorch forward and autograd results."""

import argparse
import math
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--dtype", choices=("float", "double"), required=True)
    args = parser.parse_args()
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        print("[SKIP] PyTorch is not installed in", sys.executable)
        return 77

    if args.backend == "gpu" and not torch.cuda.is_available():
        print("[SKIP] PyTorch CUDA is unavailable")
        return 77

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260916)
    dtype = torch.float32 if args.dtype == "float" else torch.float64
    device = "cuda:0" if args.backend == "gpu" else "cpu"
    atol, rtol = (2e-6, 2e-5) if args.dtype == "float" else (1e-12, 1e-10)
    names = ("relu", "relu_backward", "sigmoid", "sigmoid_backward")
    errors = dict.fromkeys(names, 0.0)
    cases = [(), (0, 3), (1,), (9,), (2, 3), (257,), (100003,)]

    for shape in cases:
        n = math.prod(shape)
        x_host = torch.randn(shape, dtype=dtype, generator=generator) * 4
        g_host = torch.randn(shape, dtype=dtype, generator=generator) * 2
        if n:
            special = torch.tensor(
                [-100, -8, -2, -0.0, 0, 0.25, 2, 8, 100], dtype=dtype
            )
            length = min(n, len(special))
            x_host.reshape(-1)[:length] = special[:length]
        payload = " ".join(map(str, [len(shape), *shape])) + "\n"
        payload += " ".join(map(repr, x_host.reshape(-1).tolist())) + "\n"
        payload += " ".join(map(repr, g_host.reshape(-1).tolist())) + "\n"
        result = subprocess.run(
            [args.driver, args.backend, args.dtype],
            input=payload,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 77:
            print("[SKIP] C++ CUDA unavailable:", result.stderr.strip())
            return 77
        if result.returncode:
            raise RuntimeError(f"C++ driver failed for {shape}: {result.stderr}")
        values = [float(value) for value in result.stdout.split()]
        if len(values) != 4 * n:
            raise AssertionError(
                f"Expected {4 * n} output values, received {len(values)}"
            )

        x = x_host.to(device).requires_grad_(True)
        g = g_host.to(device)
        relu = torch.relu(x)
        sigmoid = torch.sigmoid(x)
        # Use PyTorch autograd, not a hand-written derivative reference.
        dr = torch.autograd.grad(relu, x, grad_outputs=g)[0]
        ds = torch.autograd.grad(sigmoid, x, grad_outputs=g)[0]
        for index, (name, expected) in enumerate(zip(names, (relu, dr, sigmoid, ds))):
            actual = torch.tensor(
                values[index * n : (index + 1) * n], dtype=dtype
            ).reshape(shape)
            reference = expected.detach().cpu()
            torch.testing.assert_close(
                actual,
                reference,
                atol=atol,
                rtol=rtol,
                msg=f"{args.backend}/{args.dtype}/{shape}/{name}",
            )
            if n:
                errors[name] = max(
                    errors[name], (actual - reference).abs().max().item()
                )

    print(
        f"[PASS] PyTorch {torch.__version__}; {args.backend}/{args.dtype}; "
        f"{len(cases)} shapes; atol={atol}, rtol={rtol}"
    )
    for name, error in errors.items():
        print(f"  {name}: max_abs_error={error:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

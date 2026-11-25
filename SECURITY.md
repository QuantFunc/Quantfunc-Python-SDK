# Security Policy

We take the security of the QuantFunc Python SDK and its users seriously. All Python source code in this repository is open and auditable under the Apache-2.0 license, while the QuantFunc native engine binaries downloaded at install time are proprietary and verified before loading.

## Reporting a Vulnerability

Please report security issues **privately** — do **not** open a public GitHub issue for a suspected vulnerability.

- **Email:** `security@quantfunc.com`

Please include:

- A clear description of the issue and its security impact.
- Minimal reproduction steps (affected SDK version, OS, Python / PyTorch version, GPU model if relevant).
- Relevant logs or crash traces (redacting any API keys or credentials).

## Response Targets

- **Acknowledgement:** within 3 business days.
- **Initial triage:** within 7 business days.
- **Status updates:** at least every 14 days while open.
- **Remediation:** High/Critical issues targeted within 90 days with coordinated disclosure.

## Security Guarantees in This SDK

### API Key Handling

- Never commit API keys to source control, notebooks, or example scripts.
- Provide keys via the `QUANTFUNC_API_KEY` environment variable or save them using `quantfunc auth set-key` (interactive `getpass` prompt or `--stdin`; positional command-line arguments are rejected so keys do not leak to process listings or shell history).
- Saved credentials are stored in `~/.config/quantfunc/config.json` (parent directory `0o700`, file `0o600` on POSIX) using atomic temporary files (`O_WRONLY | O_CREAT | O_EXCL` with `O_NOFOLLOW` on POSIX).
- Exceptions, repr strings, and log messages never emit raw API keys.

### Engine Binary Verification

- Engine binaries are installed into a user-writable cache directory (`~/.cache/quantfunc/engine` or `QF_ENGINE_DIR`) and never written into `site-packages`.
- All manifest and binary fetches require HTTPS (`https://`) with strict URL validation (rejecting non-HTTPS schemes, embedded credentials, query strings, fragments, and plaintext redirects).
- Every downloaded artifact is verified against its published SHA-256 digest before atomic replacement.
- On Linux and Windows, the loader verifies host library zero-CUDA-runtime imports (ELF `DT_NEEDED` / PE import tables), Linux `.qf_pair_id` pairing, PyTorch CUDA major version alignment, and `QF_KERNEL_PLUGIN_DIR` directory/file safety (rejecting symlinks and unsafe permissions on POSIX).

## Safe Harbor

Good-faith security research conducted in accordance with this policy—avoiding privacy violations, data destruction, and service disruption—is authorized.

import os, re, sys, subprocess, shutil

# ---------- env detection ----------
def is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            v = f.read()
        return ("Microsoft" in v) or ("WSL" in v)
    except Exception:
        return "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ

def is_windows_host() -> bool:
    return os.name == "nt"

# ---------- path classification ----------
_drive_rx = re.compile(r'^(?i)([a-z]):[\\/].*')
_unc_rx   = re.compile(r'^[\\/]{2}[^\\/]+[\\/][^\\/]+(?:[\\/].*)?$')

def looks_like_windows_path(p: str) -> bool:
    return bool(_drive_rx.match(p) or _unc_rx.match(p) or ("\\" in p and not p.startswith("/")))

def looks_like_wsl_mnt(p: str) -> bool:
    # /mnt/<drive>/...
    return p.startswith("/mnt/") and len(p) >= 6 and p[5].isalpha() and p[6:7] in {"/", "\\"}

# ---------- conversions ----------
def to_wsl_path(p: str) -> str:
    """Convert Windows -> WSL (/mnt/...), or return p if already POSIX/WSL."""
    if looks_like_windows_path(p):
        if shutil.which("wslpath"):  # best: ask WSL itself
            return subprocess.check_output(["wslpath", "-u", p]).decode().strip()
        # Fallback mapping (works even outside WSL but assumes /mnt is used)
        m = _drive_rx.match(p)
        if m:
            drive = m.group(1).lower()
            rest = p[3:].replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        if _unc_rx.match(p):
            # \\Server\Share\dir\file -> /mnt/unc/Server/Share/dir/file
            q = p.lstrip("\\/")
            server, share, *rest = re.split(r"[\\/]", q)
            return "/mnt/unc/" + "/".join([server, share] + rest)
    return p.replace("\\", "/")  # already POSIX-ish

def to_windows_path(p: str) -> str:
    """Convert WSL/POSIX -> Windows (C:\\...), or return p if already Windows."""
    if looks_like_windows_path(p):
        return p
    if is_wsl() and shutil.which("wslpath"):
        return subprocess.check_output(["wslpath", "-w", p]).decode().strip()
    if looks_like_wsl_mnt(p):
        # /mnt/e/foo/bar -> E:\foo\bar
        drive = p[5].upper()
        rest = p[7:].replace("/", "\\")
        return f"{drive}:\\" + rest
    # As a last resort, just swap slashes
    return p.replace("/", "\\")

def auto_to_local(p: str) -> str:
    """
    Convert any Windows/WSL/POSIX-looking path to the *local* runtime's
    native form:
      - inside WSL  -> /mnt/... (WSL path)
      - on Windows  -> C:\...   (Windows path)
      - on Linux (non-WSL) -> leave POSIX; map Windows to /mnt/<drive>/... if sensible
    """
    if is_wsl():
        return to_wsl_path(p)
    if is_windows_host():
        return to_windows_path(p)
    # Linux (non-WSL): try a reasonable mapping for Windows paths
    wsl = to_wsl_path(p)
    return wsl
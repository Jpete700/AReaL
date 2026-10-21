"""Use the host's existing libseccomp to prohibit process creation/namespace escape."""

import ctypes
import errno
import os


def filter_fd():
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    fd = os.memfd_create("aspire-code-filter", flags=0)
    try:
        for name in (
            "clone",
            "clone3",
            "fork",
            "vfork",
            "unshare",
            "setns",
            "mount",
            "umount2",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "bpf",
            "perf_event_open",
            "keyctl",
            "add_key",
            "request_key",
            "userfaultfd",
        ):
            syscall = lib.seccomp_syscall_resolve_name(name.encode())
            if (
                syscall >= 0
                and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, syscall, 0)
                != 0
            ):
                raise RuntimeError("seccomp rule failed")
        if lib.seccomp_export_bpf(context, fd) != 0:
            raise RuntimeError("seccomp export failed")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise
    finally:
        lib.seccomp_release(context)

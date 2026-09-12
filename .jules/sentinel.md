## 2026-03-29 - Subprocess Execution Security in Clipboard Operations
**Vulnerability:** Invoking system utilities like `clip` on Windows via `subprocess.Popen` with `shell=True` introduces unnecessary command/shell injection risks and command interpreter overhead.
**Learning:** Even built-in system utility calls like `clip` or `pbcopy` do not require shell invocation (`shell=True`). Using direct executable lists avoids shell escaping pitfalls. On Linux, `shlex.split()` should be preferred over `str.split()` when parsing command string options.
**Prevention:** Always pass arguments as explicit lists to `subprocess` without `shell=True`, and parse option strings with `shlex.split()`.

## 2026-03-31 - AST Firewall Import Aliasing & Dynamic Module Loading Bypass

**Vulnerability:**
Checking AST nodes for static module names like `node.func.value.id == "os"` fails to block code using import aliases (e.g., `import os as my_os; my_os.system("...")`) or dynamic module loaders (e.g., `import importlib`).

**Learning:**
Static AST visitors must track module aliasing during `visit_Import` and `visit_ImportFrom` nodes, and prohibit dynamic module resolution libraries (`importlib`) to prevent execution sandbox escapes.

**Prevention:**
Maintain an active set of alias names (e.g., `self.os_aliases`) dynamically populated when visiting import statements, and check attribute and call operations against all recorded aliases. Prohibit `importlib` in `FORBIDDEN_MODULES`.

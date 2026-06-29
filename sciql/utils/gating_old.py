import re
import ast
import numpy as np
import jax.numpy as jnp
from typing import List, Callable, Mapping, Optional, Iterable
import jax
import ast, re, math
Array = jax.Array  # for type hints; jnp.ndarray is also fi

############################
# Operators for continuous #
############################
def union(x, y):
    """
    Performs the OR operation.
    """
    return (x + y) / 2

def sigmoid_beta(x, beta=1.0, x0=0.0):
    """
    Sigmoid operation.
    """
    return 1 / (1 + jnp.exp(-beta * (x - x0)))

def gate(x, y, beta=1.0, x0=0.0):
    """
    Performs the GATING operation.
    """
    return x + sigmoid_beta(x, beta, x0) * y

def inter(x, y, beta_x, x0, beta_y, y0):
    """
    Performs the AND operation.
    """
    return sigmoid_beta(y, beta_y, y0) * x + sigmoid_beta(x, beta_x, x0) * y

##########################
# Operators for discrete #
##########################
def union_discrete(x, y):
    """
    Performs the OR operation.
    """
    return (x + y) / 2

def inter_discrete(x, y):
    """
    Performs the AND operation.
    """
    return x * y

def gate_discrete(x, y):
    """
    Performs the GATING operation.
    """
    return (x + x * y) / 2

######################
# String evaluations #
######################
def get_function(
    x_list: List[str] = ['a_x', 'c_x', 'e_x'],
    y_list: List[str] = ['b_y', 'd_y'],
    operation_repr: str = "a_x :: c_x :: e_x",
    *,
    # Aliases for normal operators/symbols (applied after chaining rewrite)
    symbol_map: Optional[Mapping[str, str]] = None,
    # Allowlist for binary/unary operators
    allowed_ops: Iterable[str] = ("+", "-", "*", "/", "**", "//", "%"),
    allowed_unary: Iterable[str] = ("+", "-"),
    # Whitelisted functions available in expressions
    func_map: Optional[Mapping[str, Callable]] = None,
    # Named constants
    const_map: Optional[Mapping[str, float]] = None,
    # Wrap custom functions with np.vectorize (if they aren't array-aware)
    auto_vectorize: bool = False,
    # Map custom chain operator symbols to function names, e.g. {"::": "gate"}
    chain_ops: Optional[Mapping[str, str]] = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Build f(x, y) from an expression string involving named columns of x and y.
    Supports:
      - Custom operator symbols via `symbol_map`
      - Safe allowlisted function calls via `func_map`
      - N-ary left-associative *chained* infix operators via `chain_ops`,
        e.g. "a :: b :: c" -> gate(gate(a,b), c) if chain_ops={"::": "gate"}.
    """
    if symbol_map is None:
        symbol_map = {"x": "*", "×": "*", "·": "*", "^": "**", "÷": "/", "−": "-"}
    if func_map is None:
        func_map = {}
    if const_map is None:
        const_map = {"pi": float(np.pi), "e": float(np.e)}
    if chain_ops is None:
        chain_ops = {}

    if auto_vectorize and func_map:
        func_map = {name: np.vectorize(fn) for name, fn in func_map.items()}

    # ---------- helpers: chaining rewrite ----------
    def _find_matching_paren(s: str, i: int) -> int:
        """Return index of matching ')' for s[i]=='(' (no strings supported)."""
        depth = 1
        j = i + 1
        while j < len(s):
            c = s[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return j
            j += 1
        raise ValueError("Unbalanced parentheses in expression")

    def _fold_chain_top_level(s: str, symbol: str, fname: str) -> str:
        """Split s at top-level occurrences of `symbol` and fold left using fname."""
        parts = []
        i = 0
        depth = 0
        last = 0
        L = len(s)
        symL = len(symbol)
        while i < L:
            c = s[i]
            if c == "(":
                depth += 1
                i += 1
            elif c == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unbalanced parentheses in expression")
                i += 1
            elif depth == 0 and s.startswith(symbol, i):
                parts.append(s[last:i].strip())
                i += symL
                last = i
            else:
                i += 1
        parts.append(s[last:].strip())
        # No split -> no change
        if len(parts) <= 1:
            return s
        # Build left-associative nesting: f(f(p0,p1), p2) ...
        acc = parts[0]
        for p in parts[1:]:
            acc = f"{fname}({acc}, {p})"
        return acc

    def _rewrite_chains(expr: str) -> str:
        """Recursively rewrite all chain operators (outside & inside parentheses)."""
        # First, recursively process inside parentheses
        i = 0
        out = []
        while i < len(expr):
            if expr[i] == "(":
                j = _find_matching_paren(expr, i)
                inner = _rewrite_chains(expr[i+1:j])
                out.append("(" + inner + ")")
                i = j + 1
            else:
                out.append(expr[i])
                i += 1
        s = "".join(out)
        # Then, apply each chain operator at top-level
        for sym, fname in chain_ops.items():
            s = _fold_chain_top_level(s, sym, fname)
        return s

    # ---------- symbol substitution (after chaining rewrite) ----------
    def _apply_symbol_map(expr: str) -> str:
        expr = expr.strip()
        for k, v in symbol_map.items():
            if k.isidentifier():
                expr = re.sub(rf"\b{re.escape(k)}\b", v, expr)
            else:
                expr = expr.replace(k, v)
        return expr

    # 1) rewrite custom chain operators -> nested function calls
    expr = _rewrite_chains(operation_repr)
    # 2) apply symbol aliases (x -> *, ^ -> **, etc.)
    expr = _apply_symbol_map(expr)

    # ---------- parse & validate AST ----------
    node = ast.parse(expr, mode="eval")

    binop_map = {
        "+": ast.Add, "-": ast.Sub, "*": ast.Mult, "/": ast.Div,
        "**": ast.Pow, "//": ast.FloorDiv, "%": ast.Mod
    }
    unary_map = {"+": ast.UAdd, "-": ast.USub}

    allowed_bin_nodes = tuple(binop_map[o] for o in allowed_ops if o in binop_map)
    allowed_un_nodes = tuple(unary_map[o] for o in allowed_unary if o in unary_map)

    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, ast.Name, ast.Constant,
        ast.Tuple, ast.List, ast.Call
    ) + allowed_bin_nodes + allowed_un_nodes

    for n in ast.walk(node):
        if isinstance(n, (ast.Attribute, ast.Subscript, ast.Compare, ast.BoolOp,
                          ast.Lambda, ast.IfExp, ast.Dict, ast.Set,
                          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise ValueError("Only arithmetic, names, and allowed function calls are permitted.")
        if isinstance(n, ast.BinOp) and not isinstance(n.op, allowed_bin_nodes):
            raise ValueError(f"Binary operator {type(n.op).__name__} not allowed. Allowed: {set(allowed_ops)}")
        if isinstance(n, ast.UnaryOp) and not isinstance(n.op, allowed_un_nodes):
            raise ValueError(f"Unary operator {type(n.op).__name__} not allowed. Allowed: {set(allowed_unary)}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ValueError("Only plain function names are allowed in calls.")
            fname = n.func.id
            if fname not in func_map:
                raise ValueError(f"Function '{fname}' is not allowed. Allowed: {sorted(func_map)}")
            if n.keywords:
                raise ValueError("Keyword arguments are not allowed in function calls.")
            for a in n.args:
                if isinstance(a, ast.Starred):
                    raise ValueError("Starred arguments are not allowed in function calls.")
        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Disallowed AST element: {type(n).__name__}")

    used_names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    function_names = set(func_map.keys())
    allowed_var_names = set(x_list) | set(y_list) | set(const_map.keys())
    extraneous = (used_names - function_names) - allowed_var_names
    if extraneous:
        raise ValueError(f"Unknown variable/constant name(s): {sorted(extraneous)}. "
                         f"Allowed variables: {sorted(allowed_var_names)}; "
                         f"functions: {sorted(function_names)}")

    code = compile(node, filename="<operation_repr>", mode="eval")

    # ---------- build the callable ----------
    def function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[-1] != len(x_list):
            raise ValueError(f"x.shape[-1] must be {len(x_list)} for {x_list}, got {x.shape[-1]}")
        if y.shape[-1] != len(y_list):
            raise ValueError(f"y.shape[-1] must be {len(y_list)} for {y_list}, got {y.shape[-1]}")

        ctx = {name: x[..., i] for i, name in enumerate(x_list)}
        ctx.update({name: y[..., j] for j, name in enumerate(y_list)})
        ctx.update(const_map)
        ctx.update(func_map)

        return eval(code, {"__builtins__": {}}, ctx)

    return function
import re
import jax
import numpy as np
import jax.numpy as jnp
from typing import List, Mapping, Optional, Tuple, Union

ArrayLike = Union[np.ndarray, jnp.ndarray, float]

########################################
# Operators (continuous and discrete)  #
########################################

def union(x: ArrayLike, y: ArrayLike):
    """Continuous/discrete OR (as provided): (x + y) / 2"""
    return (x + y) / 2


def sigmoid_beta(x, beta: float = 1.0, x0: float = 0.0):
    return 1 / (1 + jnp.exp(-beta * (x - x0)))


def gate_cont(x: ArrayLike, y: ArrayLike, *, beta: float = 1.0, x0: float = 0.0):
    """Continuous gate: x + σ_β(x-x0) * y"""
    return x + sigmoid_beta(x, beta, x0) * y


def inter_cont(x: ArrayLike, y: ArrayLike, *, beta_x: float, x0: float, beta_y: float, y0: float):
    """Continuous inter (AND): σ_{βy}(y-y0)*x + σ_{βx}(x-x0)*y"""
    return sigmoid_beta(y, beta_y, y0) * x + sigmoid_beta(x, beta_x, x0) * y


def inter_discrete(x: ArrayLike, y: ArrayLike):
    return x * y


def gate_discrete(x: ArrayLike, y: ArrayLike):
    return (x + x * y) / 2


########################################
# Parser for '+', '*', '>' with custom #
# precedence and associativity         #
########################################

Token = Tuple[str, str]  # (type, value)

TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_]\w*)|([()+*>]))")
ALLOWED_OPS = {"+", "*", ">"}

# Binding powers (Pratt parser): '*' > '+' > '>' ; '>' is right-associative
BP = {
    "*": (30, 31),   # left-assoc
    "+": (20, 21),   # left-assoc
    ">": (10, 10),   # right-assoc
}


def tokenize(s: str):
    pos = 0
    L = len(s)
    while pos < L:
        m = TOKEN_RE.match(s, pos)
        if not m:
            raise ValueError(f"Unexpected token near: {s[pos:pos+20]!r}")
        num, name, sym = m.groups()
        pos = m.end()
        if num is not None:
            yield ("NUM", num)
        elif name is not None:
            yield ("NAME", name)
        elif sym is not None:
            if sym in "+*>":
                yield ("OP", sym)
            elif sym == "(":
                yield ("LP", sym)
            elif sym == ")":
                yield ("RP", sym)
            else:
                raise ValueError(f"Operator '{sym}' is not allowed. Use only '+', '*', '>' and parentheses.")
    yield ("EOF", "")


class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, ttype: str) -> Token:
        t = self.advance()
        if t[0] != ttype:
            raise ValueError(f"Expected {ttype}, got {t[0]} ({t[1]!r})")
        return t

    def parse(self):
        node = self.parse_expr(0)
        if self.peek()[0] != "EOF":
            raise ValueError(f"Unexpected trailing input at token {self.peek()}")
        return node

    def parse_expr(self, min_bp: int):
        # nud / primary
        ttype, tval = self.advance()
        if ttype == "NAME":
            left = ("var", tval)
        elif ttype == "NUM":
            left = ("num", float(tval))
        elif ttype == "LP":
            left = self.parse_expr(0)
            self.expect("RP")
        else:
            raise ValueError(f"Unexpected token {ttype} ({tval!r}) at start of expression")

        # led / infix loop
        while True:
            ttype, tval = self.peek()
            if ttype != "OP":
                break
            op = tval
            lbp, rbp = BP[op]
            if lbp < min_bp:
                break
            self.advance()  # consume op
            right = self.parse_expr(rbp)
            left = ("op", op, left, right)
        return left


########################################
# Tree evaluator                        #
########################################

class Evaluator:
    def __init__(self, *, mode: str,
                 gate_params: Optional[Mapping[str, float]] = None,
                 inter_params: Optional[Mapping[str, float]] = None):
        if mode not in ("continuous", "discrete"):
            raise ValueError("mode must be 'continuous' or 'discrete'")
        self.mode = mode
        self.gate_params = {"beta": 1.0, "x0": 0.0}
        if gate_params:
            self.gate_params.update(gate_params)
        # defaults for inter continuous
        self.inter_params = {"beta_x": 1.0, "x0": 0.0, "beta_y": 1.0, "y0": 0.0}
        if inter_params:
            self.inter_params.update(inter_params)

    def eval(self, node, ctx: Mapping[str, ArrayLike]):
        ntype = node[0]
        if ntype == "num":
            return node[1]
        if ntype == "var":
            name = node[1]
            if name not in ctx:
                raise ValueError(f"Unknown variable '{name}'.")
            return ctx[name]
        if ntype == "op":
            _, op, left, right = node
            a = self.eval(left, ctx)
            b = self.eval(right, ctx)
            if op == "+":
                return union(a, b)
            if op == "*":
                if self.mode == "continuous":
                    return inter_cont(a, b, **self.inter_params)
                else:
                    return inter_discrete(a, b)
            if op == ">":
                if self.mode == "continuous":
                    return gate_cont(a, b, **self.gate_params)
                else:
                    return gate_discrete(a, b)
            raise AssertionError("unreachable op")
        raise AssertionError("unreachable node type")


########################################
# Public API: get_function              #
########################################

def get_function(
    x_list: List[str],
    y_list: List[str],
    operation_repr: str,
    *,
    mode: str = "continuous",  # or "discrete"
    gate_params: Optional[Mapping[str, float]] = None,
    inter_params: Optional[Mapping[str, float]] = None,
    const_map: Optional[Mapping[str, float]] = None,
):
    """
    Build f(x, y) from `operation_repr` using ONLY the operators:
      '+' := union, '*' := inter, '>' := gate

    Precedence: '*' > '+' > '>'
    Associativity: '>' is right-associative; '+', '*' are left-associative.

    Parameters
    ----------
    x_list, y_list : names for the last-axis columns in x and y
    operation_repr : expression string using names, numbers, parentheses, and '+', '*', '>'
    mode           : 'continuous' or 'discrete' (chooses operator variants)
    gate_params    : dict for continuous gate, keys: {'beta', 'x0'}
    inter_params   : dict for continuous inter, keys: {'beta_x','x0','beta_y','y0'}
    const_map      : optional named numeric constants (e.g., {'pi': 3.14159})

    Returns
    -------
    function(x, y) -> array broadcastable to leading batch dims; x.shape[-1]==len(x_list), y.shape[-1]==len(y_list)
    """
    if const_map is None:
        const_map = {"pi": float(np.pi), "e": float(np.e)}

    # 1) Parse once
    tokens = tokenize(operation_repr)
    tree = Parser(tokens).parse()

    # 2) Build evaluator with chosen semantics
    evaluator = Evaluator(mode=mode, gate_params=gate_params, inter_params=inter_params)

    # 3) Build callable
    def fn(x, y):
        # do NOT coerce to numpy so JAX works; just shape-check
        if x.shape[-1] != len(x_list):
            raise ValueError(f"x.shape[-1] must be {len(x_list)} for {x_list}, got {x.shape[-1]}")
        if y.shape[-1] != len(y_list):
            raise ValueError(f"y.shape[-1] must be {len(y_list)} for {y_list}, got {y.shape[-1]}")
        ctx = {name: x[..., i] for i, name in enumerate(x_list)}
        ctx.update({name: y[..., j] for j, name in enumerate(y_list)})
        ctx.update(const_map)
        return evaluator.eval(tree, ctx)

    return fn


# ------------------
# Quick self-test
# ------------------
if __name__ == "__main__":

    fn = get_function(
        x_list=['a_x', 'c_x', 'e_x'],
        y_list=['b_y', 'd_y'],
        operation_repr="a_x > c_x > e_x",
        mode="continuous",
        gate_params={"beta": 1.0, "x0": 0.0},
        inter_params={"beta_x": 1.5, "x0": 0.0, "beta_y": 1.5, "y0": 0.0},
    )

    # ----- Make a batch -----
    N = 256
    # Suppose you have each feature as a (N,) array:
    a_x = jnp.linspace(-2, 2, N)
    c_x = jnp.sin(jnp.linspace(0, jnp.pi, N))
    e_x = jnp.ones(N)
    b_y = jnp.full((N,), 0.5)
    d_y = jnp.zeros((N,))

    # Pack into (N, num_features) as the function expects
    x = jnp.stack([a_x, c_x, e_x], axis=-1)  # (256, 3)
    y = jnp.stack([b_y, d_y], axis=-1)       # (256, 2)

    # ----- Option A: no vmap (already vectorized across the leading batch dim) -----
    outA = fn(x, y)                 # shape (256,)

    # JIT if you like&
    outA_jit = jax.jit(fn)(x, y)    # shape (256,)

    # ----- Option B: per-example mapping with vmap -----
    # Treat fn as operating on a single example: (3,) and (2,) → ()
    fn_vmapped = jax.vmap(fn, in_axes=(0, 0))   # vectorize over the first axis of x and y
    outB = fn_vmapped(x, y)                      # shape (256,)

    # vmap + jit together
    fast = jax.jit(jax.vmap(fn, in_axes=(0, 0)))
    outB_jit = fast(x, y)                        # shape (256,)

    print(outB_jit)
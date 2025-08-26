import argparse
import os
import random
import sys
from typing import List, Tuple
import numpy as np


# =========================
# Protected math 
# =========================
EPS = 1e-9

def pdiv(a, b):
    den = np.where(np.abs(b) < EPS, EPS, b)
    return a / den

def plog(a):
    return np.log(np.abs(a) + EPS)

def psqrt(a):
    return np.sqrt(np.abs(a))

def pexp(a):
    return np.exp(np.clip(a, -20.0, 20.0))

def ppow(a, b):
    return np.power(np.clip(a, -50.0, 50.0), np.clip(b, -3.0, 3.0))


# =========================
# Primitive set
# =========================
PRIMS = {
    "add":  dict(func=np.add,  arity=2, repr="+"),
    "sub":  dict(func=np.subtract, arity=2, repr="-"),
    "mul":  dict(func=np.multiply, arity=2, repr="*"),
    "div":  dict(func=pdiv, arity=2, repr="pdiv"),    
    "pow":  dict(func=ppow, arity=2, repr="ppow"),     
    "sin":  dict(func=np.sin, arity=1, repr="np.sin"),
    "cos":  dict(func=np.cos, arity=1, repr="np.cos"),
    "tanh": dict(func=np.tanh, arity=1, repr="np.tanh"),
    "exp":  dict(func=pexp, arity=1, repr="pexp"),
    "log":  dict(func=plog, arity=1, repr="plog"),
    "sqrt": dict(func=psqrt, arity=1, repr="psqrt"),
    "abs":  dict(func=np.abs, arity=1, repr="np.abs"),
}

# =========================
# GP Node
# =========================
class Node:
    def __init__(self, content: str, children: List["Node"] = None):
        self.content = content  
        self.children = children if children is not None else []

    def is_terminal(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        if self.is_terminal():
            return 1
        return 1 + max(ch.depth() for ch in self.children)

    def clone(self) -> "Node":
        return Node(self.content, [c.clone() for c in self.children])

    def flatten(self) -> List["Node"]:
        out = [self]
        for c in self.children:
            out.extend(c.flatten())
        return out

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (n_vars, n_samples)
        returns shape: (n_samples,)
        """
        if self.is_terminal():
            if isinstance(self.content, str) and self.content.startswith("x["):
                idx = int(self.content[2:-1])
                return x[idx]
            try:
                val = float(self.content)
            except Exception:
                val = 0.0
            return np.full(x.shape[1], val)
        prim = PRIMS[self.content]
        args = [ch.eval(x) for ch in self.children]
        y = prim["func"](*args)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return y

    def __str__(self) -> str:
        """
        Build printable expression (used for export).
        Binary: infix for +,-,* ; function-call for pdiv, ppow
        Unary: func(arg)
        Terminals: as-is
        """
        if self.is_terminal():
            return str(self.content)

        prim = PRIMS[self.content]
        r = prim["repr"]
        if prim["arity"] == 1:
            return f"{r}({self.children[0]})"
        elif prim["arity"] == 2:
            if r in {"+", "-", "*"}:
                return f"({self.children[0]} {r} {self.children[1]})"
            else:
                return f"{r}({self.children[0]}, {self.children[1]})"
        return f"{self.content}(" + ", ".join(str(c) for c in self.children) + ")"


# =========================
# Tree generators and operators
# =========================
def generate_random_tree(max_depth: int, n_vars: int, curr: int = 0) -> Node:
    if curr >= max_depth - 1 or (curr > 0 and random.random() < 0.25):
        if random.random() < 0.6:
            vidx = random.randint(0, n_vars - 1)
            return Node(f"x[{vidx}]")
        else:
            c = random.uniform(-10.0, 10.0)
            return Node(f"{c:.4f}")

    # operator
    op_key = random.choice(list(PRIMS.keys()))
    arity = PRIMS[op_key]["arity"]
    children = [generate_random_tree(max_depth, n_vars, curr + 1) for _ in range(arity)]
    return Node(op_key, children)

def mutate_point(node: Node, n_vars: int, pmut: float = 0.1) -> Node:
    node = node.clone()
    if random.random() < pmut:
        if node.is_terminal():
            if random.random() < 0.5:
                vidx = random.randint(0, n_vars - 1)
                return Node(f"x[{vidx}]")
            else:
                return Node(f"{random.uniform(-10, 10):.4f}")
        else:
            arity = PRIMS[node.content]["arity"]
            cand = [k for k, v in PRIMS.items() if v["arity"] == arity]
            node.content = random.choice(cand)
            return node
    # recurse into a random child
    if not node.is_terminal():
        i = random.randrange(len(node.children))
        node.children[i] = mutate_point(node.children[i], n_vars, pmut + 0.05)
    return node

def mutate_subtree(node: Node, max_depth: int, n_vars: int, pmut: float = 0.1, curr: int = 0) -> Node:
    node = node.clone()
    if random.random() < pmut:
        return generate_random_tree(max_depth - curr, n_vars, 0)
    if not node.is_terminal():
        i = random.randrange(len(node.children))
        node.children[i] = mutate_subtree(node.children[i], max_depth, n_vars, pmut + 0.05, curr + 1)
    return node

def mutate(node: Node, max_depth: int, n_vars: int) -> Node:
    if random.random() < 0.5:
        return mutate_point(node, n_vars)
    else:
        return mutate_subtree(node, max_depth, n_vars)

def swap_nodes(a: Node, b: Node):
    a.content, b.content = b.content, a.content
    a.children, b.children = b.children, a.children

def crossover(p1: Node, p2: Node, max_depth: int) -> Tuple[Node, Node]:
    c1, c2 = p1.clone(), p2.clone()
    n1 = c1.flatten()
    n2 = c2.flatten()
    for _ in range(16):  # try a few times to respect depth limit
        a = random.choice(n1)
        b = random.choice(n2)
        swap_nodes(a, b)
        if c1.depth() <= max_depth and c2.depth() <= max_depth:
            return c1, c2
        # revert if too deep
        swap_nodes(a, b)
    return p1.clone(), p2.clone()


# =========================
# Fitness and selection
# =========================
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    if y_pred.shape != y_true.shape:
        return float("inf")
    err = y_true - y_pred
    err = np.nan_to_num(err, nan=0.0, posinf=1e6, neginf=-1e6)
    return float(np.mean(err * err))

def fitness(ind: Node, x: np.ndarray, y: np.ndarray) -> float:
    try:
        y_pred = ind.eval(x)
        return mse(y, y_pred)
    except Exception:
        return float("inf")

def tournament(pop: List[Node], fits: List[float], k: int = 7) -> Node:
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fits[i])
    return pop[best]


# =========================
# Evolution
# =========================
def evolve(x: np.ndarray,
           y: np.ndarray,
           pop_size: int,
           generations: int,
           max_depth: int,
           elitism: float,
           seed: int = 42) -> Tuple[Node, float]:
    random.seed(seed)
    np.random.seed(seed)

    n_vars = x.shape[0]
    population = [generate_random_tree(max_depth, n_vars) for _ in range(pop_size)]
    best_ind = None
    best_fit = float("inf")

    for gen in range(generations):
        fits = [fitness(ind, x, y) for ind in population]
        # track best
        i_best = int(np.argmin(fits))
        if fits[i_best] < best_fit:
            best_fit = fits[i_best]
            best_ind = population[i_best].clone()

        # progress print
        if gen % 50 == 0 or gen == generations - 1:
            print(f"[Gen {gen:5d}] best_fitness={best_fit:g} depth={best_ind.depth() if best_ind else 0}")

        # elitism
        elite_count = max(1, int(elitism * pop_size))
        elite_idx = list(np.argsort(fits))[:elite_count]
        new_pop = [population[i].clone() for i in elite_idx]

        # fill rest with crossover/mutation
        while len(new_pop) < pop_size:
            if random.random() < 0.6 and len(population) >= 2:
                p1 = tournament(population, fits)
                p2 = tournament(population, fits)
                c1, c2 = crossover(p1, p2, max_depth)
                if random.random() < 0.3:
                    c1 = mutate(c1, max_depth, n_vars)
                if random.random() < 0.3 and len(new_pop) + 1 < pop_size:
                    c2 = mutate(c2, max_depth, n_vars)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)
            else:
                p = tournament(population, fits)
                c = mutate(p, max_depth, n_vars)
                new_pop.append(c)

        population = new_pop

    return best_ind, best_fit


# =========================
# Export
# =========================
EXPORT_HEADER = """#generated by train_and_export.py
# Student: s316628
import numpy as np

EPS = 1e-9
def _sanitize(y: np.ndarray) -> np.ndarray:
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

def pdiv(a, b):
    den = np.where(np.abs(b) < EPS, EPS, b)
    return a / den

def plog(a):
    return np.log(np.abs(a) + EPS)

def psqrt(a):
    return np.sqrt(np.abs(a))

def pexp(a):
    return np.exp(np.clip(a, -20.0, 20.0))

def ppow(a, b):
    return np.power(np.clip(a, -50.0, 50.0), np.clip(b, -3.0, 3.0))
"""

def emit_function(fid: int, expr: str) -> str:
    return (
        f"\n\ndef f{fid}(x: np.ndarray) -> np.ndarray:\n"
        f"    return _sanitize({expr})\n"
    )


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--problems", type=str, required=True, help="comma separated ids (e.g., 1,2,5)")
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--elitism", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="s316628.py")
    return ap.parse_args()

def load_problem(data_dir: str, pid: int) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(data_dir, f"problem_{pid}.npz")
    data = np.load(path)
    x = data["x"]   # (n_vars, n_samples)
    y = data["y"]   # (n_samples,)
    return x, y

def main():
    args = parse_args()
    probs = [int(p.strip()) for p in args.problems.split(",") if p.strip()]

    exported = EXPORT_HEADER
    results = {}

    for pid in probs:
        print(f"\n== Training problem {pid} ==")
        x, y = load_problem(args.data_dir, pid)
        print(f"shape x={x.shape}, y={y.shape} | pop={args.pop}, gens={args.gens}, depth={args.max_depth}, elit={args.elitism} | resume=False")

        best, best_fit = evolve(
            x=x,
            y=y,
            pop_size=args.pop,
            generations=args.gens,
            max_depth=args.max_depth,
            elitism=args.elitism,
            seed=args.seed + pid,  
        )

        expr = str(best) if best is not None else "np.zeros_like(x[0])"
        exported += emit_function(pid, expr)
        results[pid] = best_fit
        print(f"[problem {pid}] best_fitness={best_fit:g}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(exported)

    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()

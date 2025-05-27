import time, contextlib, collections

class Profiler:
    """Tiny hierarchical wall-clock profiler."""
    def __init__(self):
        self.stack  = []
        self.totals = collections.defaultdict(float)

    @contextlib.contextmanager
    def track(self, key: str):
        start = time.perf_counter()
        self.stack.append(key)
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            self.totals[key] += dt
            self.stack.pop()

    def report(self, prefix=""):
        print(prefix + "  ".join(f"{k}:{t*1e3:6.1f} ms"
                                 for k,t in self.totals.items()))
        self.totals.clear()
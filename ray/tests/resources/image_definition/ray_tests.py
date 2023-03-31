import ray
from collections import Counter
import platform

ray.init('ray://kuberay-cluster-test-head-svc:10001')
print("Connected to ray cluster") 
print("Running example ray job ...")

@ray.remote
def f(x):
    t = sum(list(range(100000)))
    return x + (platform.node(), )

out = Counter(ray.get([f.remote(()) for _ in range(1000)]))

assert len(dict(out).keys()) == 2

print("Simple tests passed")

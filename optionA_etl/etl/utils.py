import time, os
from contextlib import contextmanager
def env_str(name: str, default: str = ""): v=os.getenv(name); return v if v is not None else default
def env_int(name: str, default: int=0): 
    v=os.getenv(name)
    try: 
        return int(v) if v is not None else default
    except: 
        return default
@contextmanager
def rate_limiter(max_per_sec:int):
    interval = 1.0 / float(max_per_sec if max_per_sec>0 else 1); last=[0.0]
    yield lambda: _sleep_until_next(interval,last)
def _sleep_until_next(interval,last):
    import time as _t; now=_t.time(); el=now-last[0]; 
    if el<interval: _t.sleep(interval-el); last[0]=_t.time()

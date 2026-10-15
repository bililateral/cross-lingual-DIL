from __future__ import annotations

def projection(m: dict) -> dict:
    keys = ("t_A", "t_B", "v", "h", "L", "s_F", "r_F", "s_I", "r_I", "W_F", "W_I")
    if any(not np.isfinite(m[k]) or m[k] <= 0 for k in keys):
        raise ValueError("Actual positive finite H measurements required")
    train = 288*(m["t_A"]+m["t_B"])
    forward = 1446*m["v"]+36*m["h"]
    io = m["L"]+2*(m["s_F"]+m["r_F"]+m["s_I"]+m["r_I"])
    seconds = 1.25*(train+forward+io+1800)
    size = math.ceil(1.1*(m["W_F"]+2*m["W_I"]+2*2**30))
    return dict(train_seconds=train, forward_seconds=forward, io_seconds=io, unmeasured_allowance_seconds=1800,
                projected_total_seconds=seconds, projected_peak_output_bytes=size,
                within_budget=seconds <= 28800 and size <= 10*2**30)

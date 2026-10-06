import subprocess
SETS = [("mushroom", 24), ("iris", 61), ("car", 40978), ("dresses-sales", 23381), ("letter", 6)]
log = open("/marimo/limix_rem.log", "w")
for name, oid in SETS:
    for seed in [0, 1, 2]:
        r = subprocess.run(["python", "/marimo/limix_one.py", name, str(oid), str(seed)],
                           capture_output=True, text=True, cwd="/marimo")
        tail = (r.stdout + r.stderr)[-300:].strip().splitlines()
        last = tail[-1] if tail else "no output"
        print(name, seed, "rc=", r.returncode, last, flush=True)
        log.write(name + " " + str(seed) + " rc=" + str(r.returncode) + " " + last +chr(10))
        log.flush()
print("REMAINING DONE", flush=True)

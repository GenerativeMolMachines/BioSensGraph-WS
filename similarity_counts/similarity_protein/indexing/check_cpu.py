import os

print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
print("os.cpu_count():", os.cpu_count())
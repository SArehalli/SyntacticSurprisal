with open("template.sh", "r") as template_f:
    template = template_f.read()

model_nums = range(4)
evalsets = ["ClassicGP", "filler"]
slurm_commands = []

for e in evalsets: 
    for m in model_nums:
        with open("eval_{}_m{}.sbash".format(e, m), "w") as out_f:
            out_f.write(template.format(name="eval_{}_m{}".format(e, m), evalset=e, model=m))
        slurm_commands.append("sbatch ./scripts/eval_{}_m{}.sbash".format(e, m))

with open("run_all.sh", "w") as runall_f:
    runall_f.write("#!/bin/bash\n" + "\n".join(slurm_commands))


#!/bin/bash
set -e

cd "$HOME/_scratch/sinkhorn_project"

module load slurm/15.08.1
module load openmpi/4.0.2-gcc

mkdir -p results_new

N=4096
ITER=300

for p in 1 2 4 8; do
    for r in 1 2 3; do

        # Первое измерение для одного процесса уже выполнено
        if [ "$p" -eq 1 ] && [ "$r" -eq 1 ]; then
            continue
        fi

        echo "Submitting N=$N P=$p repeat=$r"

        sbatch -p test -N 1 -n "$p" \
            --time=00:10:00 \
            --output="$PWD/results_new/N${N}_p${p}_r${r}.out" \
            --error="$PWD/results_new/N${N}_p${p}_r${r}.err" \
            ompi ./sinkhorn_mpi "$N" "$ITER" 0
    done
done

echo "All jobs submitted."

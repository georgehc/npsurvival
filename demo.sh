set -e

declare -a methods=("cdfreg_knn_weighted" "coxph" "kernel" "knn_weighted" "rsfann" "rsf")

mkdir -p transcripts

for method in "${methods[@]}"
do
    python demo_${method}.py config.ini > transcripts/demo_${method}.txt
done

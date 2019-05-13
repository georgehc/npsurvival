set -e

declare -a methods=("cdfreg_knn_weighted" "coxph" "kernel" "knn_weighted" "rsfann" "rsf")

mkdir -p transcripts

for method in "${methods[@]}"
do
    python demo_${method}.py config_cum_haz.ini > transcripts/demo_${method}_cum_haz.txt
done

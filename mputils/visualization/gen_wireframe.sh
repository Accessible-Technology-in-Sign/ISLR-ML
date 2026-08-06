#!/bin/bash

. utils.sh

#################### To visualize over orientations for supplemental_gen dataset ####################
bp=0
pid=()

for dataset in ${DATASETS[@]}; do
for pt in ${PARTICIPANTS[@]}; do
    VIZ_OUTPUT=${VISUALIZATION_ROOT}/output/${dataset}
    if [[ ! -d ${VIZ_OUTPUT} ]]; then
        mkdir -p ${VIZ_OUTPUT}
    fi

    ./wireframe.py \
        -ss 20 \
        -pf ${MPUTILS_ROOT}/out/landmarks/${dataset}.parquet \
        -ip 0 -pt ${pt} -bp ${bp} -dbg &
    pid+=("$!")
    ((bp+=1))
done
done
wait "${pid[@]}"


# #################### To visualize main dataset and supplemental data (random sample) ####################
# smp_rate=0.05
# 
# bp=0
# pid=()
# for dataset in ${DATASETS[@]}; do
#     VIZ_OUTPUT=${VISUALIZATION_ROOT}/output/${dataset}
#     if [[ ! -d ${VIZ_OUTPUT} ]]; then
#         mkdir -p ${VIZ_OUTPUT}
#     fi
#     ./wireframe.py -pf ${MPUTILS_ROOT}/out/landmarks/${dataset}.parquet -sr ${smp_rate} -bp ${bp} &
#     pid+=("$!")
#     ((bp+=1))
# done
# wait "${pid[@]}"

# #################### To visualize over orientations for supplemental_gen dataset ####################
# bp=0
# pid=()
# 
# for dataset in ${DATASETS[@]}; do
# for suffix in ${ORIENTATION_SUFFIXES[@]}; do
#     VIZ_OUTPUT=${VISUALIZATION_ROOT}/output/${dataset}
#     if [[ ! -d ${VIZ_OUTPUT} ]]; then
#         mkdir -p ${VIZ_OUTPUT}
#     fi
#     ./wireframe.py \
#         -ss 20 \
#         -pf ${MPUTILS_ROOT}/out/landmarks/${dataset}.parquet \
#         -fv ${ORIGINAL_VIDEOS}/${dataset}/orientation-${suffix} \
#         -bp ${bp} -dbg &
#     pid+=("$!")
#     ((bp+=1))
# done
# done
# wait "${pid[@]}"


#!/bin/bash

. utils.sh

#################### To visualize Kaggle supplemental data ####################
# declare -a supp_pts=(93 227 161 254 2)
# 
# pid=()
# for pt in ${supp_pts[@]};
# do
#     ./wireframe.py --parquet_file ${MPUTILS_ROOT}/out/landmarks/supplemental.parquet --pt_id $pt > output/supplemental/$pt.txt 2>&1 &
#     pid+=("$!")
# done
# wait "${pid[@]}"

#################### To visualize newly generated supplemental data ####################
# declare -a first_twenty_pts=(3f8b 13e3 494d b2d1 c0df d3ab 8e3b fe96 8c4d a3d4 3a6e 3d12 f9ea 2ff7 e0f7 ed8e 51f5 a362 a6ed 0ba8) 
# declare -a second_twenty_pts=(812c 03ad a021 a442 1d72 711d a95b fa10 1bd5 6b92 5b63 bd21 1f91 917d fbb7 4ddc ab12 dbf9 99cb 39e5)
# declare -a third_twenty_pts=(4f1e 63a1 163a c82a f418 9d2b b718 39a6 4c3d 675f 9b23 9ed9 d478 f066 e3c0 fede 0a77 0bea d05c 9ff4)
# declare -a pts=(f760 7f32 80fe 19d3 6f68 a3e7 cf84 d69c 1f86 2f35 e4fa 5d33)
# declare -a pts=(3f8b 13e3 494d b2d1 c0df d3ab 8e3b fe96 8c4d a3d4 3a6e 3d12 f9ea 2ff7 e0f7 ed8e 51f5 a362 a6ed 0ba8 812c 03ad a021 a442 1d72 711d a95b fa10 1bd5 6b92 5b63 bd21 1f91 917d fbb7 4ddc ab12 dbf9 99cb 39e5 4f1e 63a1 163a c82a f418 9d2b b718 39a6 4c3d 675f 9b23 9ed9 d478 f066 e3c0 fede 0a77 0bea d05c 9ff4 f760 7f32 80fe 19d3 6f68 a3e7 cf84 d69c 1f86 2f35 e4fa 5d33)
# declare -a pts=(03ad 0a77 0ba8 0bea 13e3) 
#
# pid=()
# for pt in ${pts[@]};
# do
#     ./wireframe.py --parquet_file ${MPUTILS_ROOT}/out/landmarks/supplemental_gen.parquet --pt_id $pt > output/supplemental_gen/$pt.txt 2>&1 &
#     pid+=("$!")
# done
# wait "${pid[@]}"

#################### To visualize main dataset and supplemental data (random sample) ####################
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

#################### To visualize over orientations for supplemental_gen dataset ####################
bp=0
pid=()

for dataset in ${DATASETS[@]}; do
for suffix in ${ORIENTATION_SUFFIXES[@]}; do
    VIZ_OUTPUT=${VISUALIZATION_ROOT}/output/${dataset}
    if [[ ! -d ${VIZ_OUTPUT} ]]; then
        mkdir -p ${VIZ_OUTPUT}
    fi
    ./wireframe.py \
        -ss 20 \
        -pf ${MPUTILS_ROOT}/out/landmarks/${dataset}.parquet \
        -fv ${ORIGINAL_VIDEOS}/${dataset}/orientation-${suffix} \
        -bp ${bp} -dbg &
    pid+=("$!")
    ((bp+=1))
done
done
wait "${pid[@]}"


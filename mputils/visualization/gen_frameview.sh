#!/bin/bash

. utils.sh

############################## GEN supplemental gen missing frames analysis ##############################
# python timeline_viz.py --parquet_file ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.1_0.3.parquet --metadata_file ${MPUTILS_ROOT}/out/metadata/supplemental_gen_nan_rng0.1_0.3.csv
# python timeline_viz.py --parquet_file ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.3_0.5.parquet --metadata_file ${MPUTILS_ROOT}/out/metadata/supplemental_gen_nan_rng0.3_0.5.csv
# python timeline_viz.py --parquet_file ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.5_0.7.parquet --metadata_file ${MPUTILS_ROOT}/out/metadata/supplemental_gen_nan_rng0.5_0.7.csv
# python timeline_viz.py --parquet_file ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.7_0.9.parquet --metadata_file ${MPUTILS_ROOT}/out/metadata/supplemental_gen_nan_rng0.7_0.9.csv

for dataset in ${DATASETS[@]}; do
    ./frameview.py -pf ${MPUTILS_ROOT}/out/landmarks/${dataset}.parquet
done


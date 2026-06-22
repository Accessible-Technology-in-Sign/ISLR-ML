#!/bin/bash

#################### run on nan ranges ####################
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/ --save_parquet_file ./out/landmarks/supplemental_gen_rng0.0_0.1.parquet --sample_method nan_range --rng 0.0 0.1
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/ --save_parquet_file ./out/landmarks/supplemental_gen_rng0.1_0.3.parquet --sample_method nan_range --rng 0.1 0.3
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/ --save_parquet_file ./out/landmarks/supplemental_gen_rng0.3_0.5.parquet --sample_method nan_range --rng 0.3 0.5
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/ --save_parquet_file ./out/landmarks/supplemental_gen_rng0.5_0.7.parquet --sample_method nan_range --rng 0.5 0.7
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/ --save_parquet_file ./out/landmarks/supplemental_gen_rng0.7_0.9.parquet --sample_method nan_range --rng 0.7 0.9
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/ --save_parquet_file ./out/landmarks/supplemental_gen_rng0.9_1.0.parquet --sample_method nan_range --rng 0.9 1.0

#################### run on supplemental gen and main datasets ####################
python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json --h5_loc ./data/h5_default/supplemental_gen --save_parquet_file ./out/landmarks/supplemental_gen.parquet
python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/daun_v1/metadata/daun_v1-train.json --h5_loc ./data/h5_default/main_train --save_parquet_file ./out/landmarks/main_train.parquet
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/daun_v1/metadata/daun_v1-validation.json --h5_loc ./data/h5_default/main_validation --save_parquet_file ./out/landmarks/main_validation.parquet
# python analysis.py --action_type make_parquet --metadata_file /data/sign_language_videos/fingerspelling_videos/daun_v1/metadata/daun_v1-test.json --h5_loc ./data/h5_default/main_test --save_parquet_file ./out/landmarks/main_test.parquet

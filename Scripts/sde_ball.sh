############################################################### Collect Support Data
#CUDA_VISIBLE_DEVICES=1 python ball_collect_data.py \
#--data_name Sorting_SDE_Support_n1e5_ball1x12_bound0.15 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python ball_collect_data.py \
#--data_name Sorting_SDE_Support_n1e6_ball1x12_bound0.15 \
#--n_samples 1000000 \
#--is_gui True \

#CUDA_VISIBLE_DEVICES=1 python ball_collect_data.py \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_samples 100000 \
#--n_box 10 \
#--wall_bound 0.3 \
#--is_gui False \

#CUDA_VISIBLE_DEVICES=1 python ball_collect_data.py \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--n_samples 100000 \
#--n_box 10 \
#--wall_bound 0.2 \

############################################################### Train Support
#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D12_SDE_Support_n1e5_ball1x12 \
#--data_name Sorting_SDE_Support_n1e5_ball1x12 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D13_SDE_Support_n1e5_ball1x12_t001 \
#--data_name Sorting_SDE_Support_n1e5_ball1x12 \
#--n_samples 100000 \

# update the networks
#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D13_SDE_Support_n1e5_ball1x12_t001_bs256_updateNN_long \
#--data_name Sorting_SDE_Support_n1e5_ball1x12 \
#--batch_size 256 \
#--n_epoches 10000 \
##--n_samples 100000 \


#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D18_SDE_Support_n1e5_ball1x10_t001_bs64 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D18_SDE_Support_n1e5_ball1x10_t001_bs64_NN_long \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name test \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D19_SDE_Support_n1e5_ball1x10_t001_bs64_updateNN \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D19_SDE_Support_n1e5_ball1x10_t001_bs64_updateNN_sigma10 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D19_SDE_Support_n1e5_ball1x10_t001_bs64_updateNN_sigma5 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.3 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D20_SDE_Support_n1e5_ball1x10_t001_bs64_updateNN_sigma5 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D20_SDE_Support_n1e5_ball1x10_t001_bs64_updateNN_sigma10 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--n_box 10 \
#--wall_bound 0.3 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D20_n1e5_10balls_bound02_oldmodel_sigma25_normalize_knn9 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 256 \
#--knn 9 \
#--sigma 25 \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D20_n1e5_10balls_bound02_oldmodel_sigma25_normalize_knn7 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 100000 \
#--knn 7 \
#--sigma 25 \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D21_n1e5_10balls_bound02_oldmodel_sigma25_unnormalize_knn7 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 100000 \
#--knn 7 \
#--sigma 25 \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
##--exp_name M3D21_n1e5_10balls_bound02_Mini_sigma25_unnormalize_knn7 \
##--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
##--batch_size 512 \
##--n_epoches 10000 \
##--knn 7 \
##--sigma 25 \
##--hidden_dim 128 \
##--embed_dim 64 \
##--arch Mini \
##--n_box 10 \
##--wall_bound 0.2 \
##--normalize False \
##--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D21_n1e5_10balls_bound02_MiniUpdate_sigma25_unnormalize_knn7 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 10000 \
#--knn 7 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.2 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.2 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.1 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.1 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=0 python Trainers/SortingSDE.py \
#--exp_name M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.08 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.08 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \


#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.3 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.3 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
##--exp_name M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.06 \
##--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
##--batch_size 512 \
##--n_epoches 3000 \
##--r 0.06 \
##--sigma 25 \
##--hidden_dim 128 \
##--embed_dim 64 \
##--arch Mini \
##--n_box 10 \
##--wall_bound 0.2 \
##--normalize False \
##--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=0 python Trainers/SortingSDE.py \
#--exp_name M3D22_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.05 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.05 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--normalize False \
#--n_samples 100000 \


#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_n1e5_10balls_bound02_Mini_sigma25_unnormalize_knn7 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 2000 \
#--knn 7 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_n1e5_10balls_bound02_Mini_sigma25_unnormalize_radius0.2 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.2 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch Mini \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_n1e5_10balls_bound02_MiniUpdate_sigma25_unnormalize_radius0.2 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--batch_size 512 \
#--n_epoches 3000 \
#--r 0.2 \
#--sigma 25 \
#--hidden_dim 128 \
#--embed_dim 64 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_wmd_debug \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 3000 \
#--lr 2e-5 \
#--r 0.4 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_wmd_debug_lr2e5_bs1024 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 3000 \
#--lr 2e-5 \
#--r 0.4 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_wmd_debug_lr2e5_bs1024_tscale01 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.1 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-5 \
#--r 0.4 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_wmd_debug_lr2e5_bs1024_tscale01_r08 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.1 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_wmd_debug_lr2e5_bs1024_tscale10_r08 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 1.0 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D23_wmd_debug_lr2e5_bs1024_tscale01_r04_doubleLayer \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.1 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.4 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniDouble \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D26_wmd_debug_lr2e5_bs1024_tscale10_r08 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 1.0 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D26_wmd_debug_lr2e5_bs1024_tscale10_r08_randommask \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 1.0 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D26_wmd_debug_lr2e5_bs1024_tscale10_r08_randommask8 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 1.0 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D27_wmd_debug_lr2e5_bs1024_tscale10_r08_randommask5_oddfunc \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 1.0 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D27_wmd_debug_lr2e5_bs1024_tscale001_r08_oddfunc \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.01 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D27_wmd_debug_lr2e5_bs1024_tscale0001_r08_oddfunc \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.001 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D27_wmd_debug_lr2e5_bs1024_tscale0005_r08_oddfunc \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.005 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D27_wmd_debug_lr2e5_bs1024_tscale001_r08_oddfunc_radius2 \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.01 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 2.0 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \

#CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
#--exp_name M3D27_wmd_debug_lr2e5_bs1024_tscale001_r08_oddfunc_3mlps \
#--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
#--tscale 0.01 \
#--normalize True \
#--batch_size 1024 \
#--n_epoches 10000 \
#--lr 2e-4 \
#--r 0.8 \
#--sigma 25 \
#--hidden_dim 256 \
#--embed_dim 128 \
#--arch MiniUpdate \
#--n_box 10 \
#--wall_bound 0.2 \
#--n_samples 100000 \


# CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
# --exp_name M3D30_r08 \
# --data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
# --tscale 0.01 \
# --normalize True \
# --batch_size 1024 \
# --n_epoches 10000 \
# --lr 2e-4 \
# --r 0.8 \
# --sigma 25 \
# --hidden_dim 256 \
# --embed_dim 128 \
# --arch MiniUpdate \
# --n_box 10 \
# --wall_bound 0.2 \
# --n_samples 100000 \


# CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
# --exp_name M3D30_r06 \
# --data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
# --tscale 0.01 \
# --normalize True \
# --batch_size 1024 \
# --n_epoches 10000 \
# --lr 2e-4 \
# --r 0.6 \
# --sigma 25 \
# --hidden_dim 256 \
# --embed_dim 128 \
# --arch MiniUpdate \
# --n_box 10 \
# --wall_bound 0.2 \
# --n_samples 100000 \


# CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
# --exp_name M3D30_r04 \
# --data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
# --tscale 0.01 \
# --normalize True \
# --batch_size 1024 \
# --n_epoches 10000 \
# --lr 2e-4 \
# --r 0.4 \
# --sigma 25 \
# --hidden_dim 256 \
# --embed_dim 128 \
# --arch MiniUpdate \
# --n_box 10 \
# --wall_bound 0.2 \
# --n_samples 100000 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M4D16_r04 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.01 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.4 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r04 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.01 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.4 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.4 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r06 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.01 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.6 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.6 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r06 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.01 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.8 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.8 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r08 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.01 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.8 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.8 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r08 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.01 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.8 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.8 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r08_scale01 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.1 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.8 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.8 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r08_scale0001 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.001 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.8 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.8 \

CUDA_VISIBLE_DEVICES=1 python Trainers/SortingSDE.py \
--exp_name M5D1_r08__scale0005 \
--data_name Sorting_SDE_Support_n1e5_ball1x10_bound0.2 \
--tscale 0.005 \
--normalize True \
--batch_size 1024 \
--n_epoches 10000 \
--lr 2e-4 \
--r 0.8 \
--sigma 25 \
--hidden_dim 256 \
--embed_dim 128 \
--arch MiniUpdate \
--n_box 10 \
--wall_bound 0.2 \
--n_samples 100000 \
--radius 0.8 \
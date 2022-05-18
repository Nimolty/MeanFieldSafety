#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r08 \
# --n_boxes 10 \
# --radius 0.8 \
# --max_vel 0.3 \
# --sup_rate 0.1
# --wall_bound 0.2
# 量化指标： Average vel err 0.431 / Mean vel err 0.1295
# Mean Collision Num 1.92 / Collision Num 480.0
# 视频结果
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r08 \
# --n_boxes 10 \
# --radius 0.8 \
# --max_vel 0.3 \
# --sup_rate 0.3
# --wall_bound 0.2
# 量化指标： Average vel err 0.982 / Mean vel err 0.382
# Mean Collision Num 8.38 / Collision Num 2095
# 视频结果：所有小球聚在了一起，无法到达指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 5 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.1
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.2
# 量化指标： 
### Average vel err: 0.12039782632589849 || Mean vel err: 0.0412823676943257###
###All delta_pos : 0.03606033697724342 || Mean delta_pos : 0.007212067488580942 || Max delta_pos : 0.012314876541495323###
### Mean Collision Num: 0.056 || Total Collision Num: 14.0 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 5 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.3
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.2
# 量化指标： 
### Average vel err: 0.13731539705158788 || Mean vel err: 0.04032422844074101###
###All delta_pos : 0.09495609253644943 || Mean delta_pos : 0.018991218879818916 || Max delta_pos : 0.047611162066459656###
###Totally safe###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.1
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.2
# 量化指标： 
### Average vel err: 0.24398525853343425 || Mean vel err: 0.05317853846748339###
###All delta_pos : 0.11175739020109177 || Mean delta_pos : 0.011175738647580147 || Max delta_pos : 0.030191674828529358###
### Mean Collision Num: 0.452 || Total Collision Num: 113.0 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.2
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.2
# 量化指标： 
### Average vel err: 0.17019870838088727 || Mean vel err: 0.03376182784903415###
###All delta_pos : 0.14926838874816895 || Mean delta_pos : 0.01492683868855238 || Max delta_pos : 0.045122090727090836###
### Mean Collision Num: 0.145 || Total Collision Num: 36.25 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.3
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.2 
# 量化指标： 
### Average vel err: 0.1502321630361693 || Mean vel err: 0.033246847810887756###
###All delta_pos : 0.8027579188346863 || Mean delta_pos : 0.08027578890323639 || Max delta_pos : 0.41892915964126587###
### Mean Collision Num: 0.146 || Total Collision Num: 36.5 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.5
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.2
# 量化指标： 
### Average vel err: 0.15206382250025113 || Mean vel err: 0.033186710741380195###
###All delta_pos : 0.9052395820617676 || Mean delta_pos : 0.09052395820617676 || Max delta_pos : 0.38507765531539917###
#Totally safe
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 30 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.1
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.28
# 量化指标： 
### Average vel err: 0.6063130806581165 || Mean vel err: 0.0852474979965857###
###All delta_pos : 2.0949294567108154 || Mean delta_pos : 0.069830983877182 || Max delta_pos : 0.7628868222236633###
### Mean Collision Num: 3.99 || Total Collision Num: 997.5 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 30 \
# --radius 0.4 \
# --max_vel 0.3 \
# --sup_rate 0.3 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.28
# 量化指标： 
### Average vel err: 0.5803625176628668 || Mean vel err: 0.03964238932214547###
###All delta_pos : 7.781487941741943 || Mean delta_pos : 0.2593829333782196 || Max delta_pos : 1.44132399559021###
### Mean Collision Num: 0.482 || Total Collision Num: 120.5 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D6_r08_scale015 \
# --n_boxes 10 \
# --radius 0.8 \
# --max_vel 0.3 \
# --sup_rate 0.1 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.20
# 量化指标： 
### Average vel err: 0.42847014294744523 || Mean vel err: 0.13583889072784586###
###All delta_pos : 0.08274920284748077 || Mean delta_pos : 0.008274920284748077 || Max delta_pos : 0.022147690877318382###
### Mean Collision Num: 2.302 || Total Collision Num: 575.5 ###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D6_r08_scale015 \
# --n_boxes 10 \
# --radius 0.8 \
# --max_vel 0.3 \
# --sup_rate 0.1 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.20
# 量化指标： 
### Average vel err: 0.5266268365547083 || Mean vel err: 0.16943406979091524###
###All delta_pos : 1.1098363399505615 || Mean delta_pos : 0.11098363250494003 || Max delta_pos : 0.7444146275520325###
### Mean Collision Num: 3.035 || Total Collision Num: 758.75 ##
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.2 \
# --sup_rate 0.3 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.20
# 量化指标： 
### Average vel err: 0.14442756309423377 || Mean vel err: 0.03565970923135026###
###All delta_pos : 0.8831984400749207 || Mean delta_pos : 0.08831984549760818 || Max delta_pos : 0.4032108783721924###
###Totally safe###
# 视频结果：到达了指定位置
#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.15 \
# --sup_rate 0.2 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.20
# 量化指标： 
### Average vel err: 0.16741926021405001 || Mean vel err: 0.03729325369333637###
###All delta_pos : 0.18022741377353668 || Mean delta_pos : 0.018022742122411728 || Max delta_pos : 0.06688100844621658###
###Totally safe###
# 视频结果：到达了指定位置

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --radius 0.4 \
# --max_vel 0.49 \
# --sup_rate 1.06 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.20
# 量化指标： 
### Average vel err: 0.20834771171495756 || Mean vel err: 0.03821383525567756###
###All delta_pos : 0.18387456238269806 || Mean delta_pos : 0.018387455493211746 || Max delta_pos : 0.0600334107875824###
###Totally safe###
# 视频结果：到达了指定位置

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 30 \
# --radius 0.4 \
# --max_vel 1.8484 \
# --sup_rate 1.05 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --wall_bound 0.7
# 量化指标： 
### Average vel err: 0.5160280951927735 || Mean vel err: 0.020035658499946485###
###All delta_pos : 0.9957493543624878 || Mean delta_pos : 0.03319164365530014 || Max delta_pos : 0.0880768671631813###
### Mean Collision Num: 0.01 || Total Collision Num: 2.5 ###
# 视频结果：到达了指定位置

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --n_boxes 10 \
# --neighbor_std 4.5 \
# 这个实验的目的是重新设置target_state的活动半径，原先的话太困难了

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D6_r04 \
# --n_boxes 60 \
# --radius 0.08 \
# --max_vel_ratio 0.685 \
# --sup_rate 0.11 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --neighbor_std 4.5
# 量化指标： 
### Average vel err: 0.02378026485644819 || Mean vel err: 0.001837816157940173###
###All delta_pos : 0.3306301236152649 || Mean delta_pos : 0.005510502029210329 || Max delta_pos : 0.020111139863729477###
### Mean Collision Num: 0.003 || Total Collision Num: 0.75 ###

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb test_final.py \
# --exp_name M5D1_r04 \
# --n_boxes 60 \
# --radius 0.08 \
# --max_vel_ratio 0.685 \
# --sup_rate 0.11 \
# --dt 1/50 \
# --pb_freq 4 \
# --duration 5\
# --neighbor_std 8
# 量化指标： 
### Average vel err: 0.37175488947906593 || Mean vel err: 0.005798921807748367###
###All delta_pos : 0.32274577021598816 || Mean delta_pos : 0.0053790961392223835 || Max delta_pos : 0.02037626877427101###
### Mean Collision Num: 0.001 || Total Collision Num: 0.25 ###



#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 10 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate 0.05 \
# --scale 2
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 30 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 60 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 90 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 10 \
# --dist_r 3 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 30 \
# --dist_r 3 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 60 \
# --dist_r 3 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 90 \
# --dist_r 3 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 8

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 10 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 0

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 30 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 0

#######################################################
# CUDA_VISIBLE_DEVICES=1 python -m ipdb Finding.py \
# --exp_name M5D1_r04 \
# --date 514
# --n_boxes 60 \
# --dist_r 5 \
# --max_vel_ratio 0.1 \
# --sup_rate_init 0.05 \
# --scale 2.
# --neighbor_std 0









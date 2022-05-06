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






















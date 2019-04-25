# stcgan

This is an unofficial implementation of CVPR 2018 "Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal"

# Q&A
1. what's the difference between natural scene and documents shadow removal?
    
# EXPERIMENT
1. stcgan_fusion.py
   ---------          ----------          ----------          ------------
   | S_img |  ----->  | STCGAN |  ----->  | N1_img |  ----->  |          |
   ---------          ----------          ----------          |          |         ---------
       |                                                      | Blending |  -----> | N_img |
       |              ----------          ----------          |          |         ---------
       ------------>  |  ACCV  |  ----->  | N2_img |  ----->  |          |
                      ----------          ----------          ------------

2. pix2pix.py
   ---------          ----------          ----------          ----------         ---------
   | S_img |  ----->  |  ACCV  |  ----->  | N1_img |  ----->  | STCGAN |  -----> | N_img |
   ---------          ----------          ----------          ----------         ---------


3. stcgan_concat.py
   ---------          ----------          ----------          ----------
   | S_img |  ----->  |  ACCV  |  ----->  | N1_img |  ----->  |        |
   ---------          ----------          ----------          |        |         ---------
       |                                                      | STCGAN |  -----> | N_img |
       |                                                      |        |         ---------
       ---------------------------------------------------->  |        |
                                                              ----------



# TODO

1. find the best model 
2. record the iteration and save it in to file, so it can be resumed when reloaded
3. plot the training loss

# Execution command
1. train model 
    ./run.sh --mode train --gpu_id 0 --epochs 500 --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size 8 --dataset_name DSRD_BW
    ./run.sh --mode train --gpu_id 0 --epochs 500 --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size 8 --dataset_name DSRD_all
2. test model
    ./run.sh --mode test --load_model --gpu_id 0 --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size_test 10 --dataset_name DSRD_aligned --model_name latest
    ./run.sh --mode test --load_model --gpu_id 0 --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size_test 10 --dataset_name DSRD_aligned
3. evaluate all model
    python src/eval_all_model.py --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size_test 10 --lrG 0.001 --lrD 0.001 --dataset_name DSRD_aligned
    python src/eval_all_model.py --root_dir /media/yslin/SSD_DATA/research/stcgan/ --batch_size_test 10 --lrG 0.001 --lrD 0.001 --dataset_name ISTD
    
4. eval result
    python src/eval.py ISTD /media/yslin/SSD_DATA/research/stcgan/processed_dataset/ISTD/result/Guo/
    python src/eval.py ISTD /media/yslin/SSD_DATA/research/stcgan/processed_dataset/ISTD/result/Yang/
    python src/eval.py ISTD /media/yslin/SSD_DATA/research/stcgan/processed_dataset/ISTD/result/Gong/
    python src/eval.py ISTD /media/yslin/SSD_DATA/research/stcgan/processed_dataset/ISTD/result/ST-CGAN/
    python src/eval.py ISTD /media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/ISTD/result/non_shadow

    python src/eval.py DSRD_aligned /media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/shadow
    python src/eval.py DSRD_aligned /media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/result/ACCV2016
    python src/eval.py DSRD_all /media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_all/result/non_shadow
    

    python src/eval.py Blender39_aligned /media/yslin/SSD_DATA/research/processed_dataset/Blender39_aligned/test/shadow
    
    <!-- python src/eval.py DSRD_text /media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_all/result/non_shadow -->
    <!-- python src/eval.py DSRD_text /media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_text/result/non_shadow -->
5. visualize grid result
    python src/displayPairDSRD.py

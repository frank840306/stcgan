import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import os


def display(titles=None, dirs=None, kwargs=[{}], img_list=None, batch_size=4):
    row = batch_size
    col = len(titles)

    
    if img_list:
        img_list = [[os.path.join(dirs[i], img) for img in img_list] for i in range(col)]
    else:
        assert(all([
            len(titles) == len(dirs)
        ] + [len(os.listdir(dirs[0])) == len(os.listdir(dirs[i])) for i in range(1, col)]
        ))
        img_list = [[fp for fp in sorted(glob.glob(os.path.join(dirs[i], '*.png')))] for i in range(col)]
    img_num = len(img_list[0])
    
    batch_num = img_num // row
    for b in range(batch_num):
        fig = plt.figure(figsize=(30, 45))
        fig.suptitle('')
        for r in range(row):
            basename = os.path.basename(img_list[0][b * row + r])
            for c in range(col): 
                print(img_list[c][b * row + r])
                img = mpimg.imread(img_list[c][b * row + r])      
                ax = fig.add_subplot(row, col, r * col + c + 1)
                if c == 0:
                    ax.set_ylabel(basename)
                if r == 0:
                    ax.set_title(titles[c])
                plt.imshow(img, **kwargs[c])
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()
    plt.close()
    return

def main():
    # # compare between dsrd_all and dsrd_extend
    # display(
    #     titles=[
    #         'shadow', 
    #         'non shadow\n(ground truth)', 
    #         'mask\n(ground truth)', 
    #         'ACCV2016', 
    #         # 'ST-CGAN\n(70%)', 
    #         'ST-CGAN\n(DSRD)',
    #         'ST-CGAN\n(DSRD extend)'

    #     ],
    #     dirs=[
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/test/shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/test/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/test/mask',
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/result/ACCV2016/',
    #         # '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_70/result/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_all/result/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_extend/result/non_shadow',
    #     ],
    #     img_list=[
    #         'S001T006N00004.png',
    #         'S001T006N00008.png',
    #         'S005T006N00005.png',
    #         'S005T006N00008.png',
    #     ],
    #     kwargs=[{}, {}, {'cmap': 'Greys_r'}, {}, {}, {}]
    # )
    # compare between gt and st-cgan (DSRD and DSRD_extend)
    # display(
    #     titles=[
    #         'shadow', 
    #         'non shadow\n(ground truth)', 
    #         'non shadow\n(ST-CGAN & DSRD)', 
    #         'non shadow\n(ST-CGAN & DSRD extend)',
    #         'mask\n(ground truth)', 
    #         'mask\n(ST-CGAN & DSRD)',
    #         'mask\n(ST-CGAN & DSRD extend)',
    #     ],
    #     dirs=[
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/test/shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/test/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_all/result/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_extend/result/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/processed_dataset/DSRD_extend/test/mask',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_all/result/mask',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_extend/result/mask',
    #     ],
    #     img_list=[
    #         'S001T006N00004.png',
    #         'S001T006N00008.png',
    #         'S005T006N00005.png',
    #         'S005T006N00008.png',
    #     ],
    #     kwargs=[{}, {}, {}, {}, {'cmap': 'Greys_r'}, {'cmap': 'Greys_r'}, {'cmap': 'Greys_r'}]
    # )
    # compare between accv2016 and st-cgan
    # display(
    #     titles=[
    #         'shadow', 
    #         'non shadow\n(ground truth)', 
    #         'mask\n(ground truth)', 
    #         'ACCV2016', 
    #         # 'ST-CGAN\n(70%)', 
    #         'ST-CGAN',
    #         'deRaindrop',
    #     ],
    #     dirs=[
    #         '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/shadow',
    #         '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/non_shadow',
    #         '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/mask',
    #         '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/result/ACCV2016/',
    #         # '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_70/result/non_shadow',
    #         '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001_randomRotation/DSRD_aligned/result/non_shadow',
    #         '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/result/deRaindrop_85000/non_shadow',
    #     ],
    #     img_list=[
    #         # 'S006T001N00014.png',
    #         'S010T001N00015.png',
    #         'S018T001N00006.png',
    #         'S026T004N00017.png',
    #         'S029T006N00003.png',
    #     ],
    #     kwargs=[{}, {}, {'cmap': 'Greys_r'}, {}, {}, {}]
    # )
    # compare between gt and st-cgan

    display(
        titles=[
            'shadow', 
            'non shadow\n(ground truth)', 
            'non shadow\n(ST-CGAN)',
            'non shadow\n(deRaindrop)', 
            'mask\n(ground truth)', 
            'mask\n(ST-CGAN)',
            'mask\n(deRaindrop)'
        ],
        dirs=[
            '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/shadow',
            '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/non_shadow',
            '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_aligned/result/non_shadow',
            '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/result/deRaindrop_85000/non_shadow',
            '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/test/mask',
            '/media/yslin/SSD_DATA/research/stcgan/task/stcgan_lrG_0.001_lrD_0.001/DSRD_aligned/result/mask',
            '/media/yslin/SSD_DATA/research/processed_dataset/DSRD_aligned/result/deRaindrop_85000/mask',
        ],
        img_list=[
            'S018T001N00003.png',
            'S022T003N00013.png',
            'S026T004N00004.png',
            'S029T006N00013.png',
        ],
        kwargs=[{}, {}, {}, {}, {'cmap': 'Greys_r'}, {'cmap': 'Greys_r'}, {'cmap': 'Greys_r'}]
    )
    

if __name__ == "__main__":
    main()
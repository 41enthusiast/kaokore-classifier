import shutil, os
import random
import glob

paths = ['../../fst-kaokore-cb', '../../fst-kaokore-2-cb']
tgt_path = '../../kaokore_imagenet_style/status/train'
subsets = ['10pct', '25pct','50pct', '100pct']
subsets_pct = [0.1, 0.25, 0.5, 1]

class_names = os.listdir(tgt_path)

for cls in class_names:
    for img_dir in paths:
        temp = glob.glob(img_dir+'/'+cls+'/[!0]*')
        #print(temp)
        orig_count = len(os.listdir(tgt_path+'/'+cls))
        random.shuffle(temp)

        for subset, subset_pct in zip(subsets, subsets_pct):
            current_tgt_path = img_dir+'-'+subset+'/'+cls
            current_src_path = temp[:int(subset_pct*orig_count)]
            print(current_tgt_path, 'Extra img count', int(subset_pct*orig_count), 'Original img count', orig_count)
            shutil.copytree(tgt_path+'/'+cls, current_tgt_path, dirs_exist_ok=True)
            for img_path in current_src_path:
                shutil.copyfile('/'.join([img_dir,cls,img_path.split('/')[-1]]), current_tgt_path+'/'+img_path.split('/')[-1])

            



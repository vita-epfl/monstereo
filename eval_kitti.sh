#!/bin/bash
cuda_device_to_use='1'

use_car='0'

epochs='100'

hidden_size='1024' 

joints_stereo='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-stereo-201021-1518.json'
joints_mono='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-201023-1410.json'
dir_ann='/data/maxime-data/outputs_openpifpaf_kitti/annotations'

joints_stereo_car='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-vehicles-stereo-201022-1536.json'
joints_mono_car='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-vehicles-201022-1537.json'
dir_ann_car='/data/maxime-data/outputs_openpifpaf_kitti/annotations_car'


model_out () {
    while read -r line; do
        id=`echo $line | cut -c 1-12`    
        if [ $id == "data/models/" ]    
        then         
        echo "$id"
        model=$line  
        echo "$model"
        fi 
    done <<< $1
}


if [ $use_car == "1" ]
then 
# train mono model
output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --dataset kitti --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset kitti --save)
model_out "$output"
model_mono="$model"

echo "$model_mono"

output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --dataset kitti --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset kitti --save)
model_out "$output"
model_stereo="$model"

echo "$model_stereo"

"Generate and evaluate the outputs"
CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m monstereo.run eval --dir_ann ${dir_ann_car} --model ${model_stereo}  --model-mono ${model_mono} --vehicles --generate

else

# train mono model
output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --dataset kitti --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --save)
model_out "$output"
model_mono="$model"

echo "$model_mono"

output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --dataset kitti --joints ${joints_stereo} --hidden_size ${hidden_size} --dataset kitti --save)
model_out "$output"
model_stereo="$model"

echo "$model_stereo"

"Generate and evaluate the outputs"
CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m monstereo.run eval --dir_ann ${dir_ann} --model ${model_stereo}  --model-mono ${model_mono} --generate


fi

#CUDA_VISIBLE_DEVICES=1 python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/00000*.png --joints_folder /home/maximebonnesoeur/monstereo/data/kitti-pifpaf/annotations_car --mode mono --model /home/maximebonnesoeur/monstereo/data/models/ms-201023-1411-vehicles.pkl --show --output_types combined --vehicles --path_gt /home/maximebonnesoeur/monstereo/data/arrays/names-kitti-vehicles-201022-1537.json --draw_box
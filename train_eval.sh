#!/bin/bash
cuda_device_to_use='0'

use_car='1'

hyp='0'

multipler='2'

eval_mode='--save --verbose'

stereo='0'

epochs='200'

hidden_size='1024' 

joints_there='0'


joints_stereo='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-stereo-201021-1518.json'
joints_mono='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-201023-1410.json' 
dir_ann='/data/maxime-data/outputs_openpifpaf_kitti/annotations'

joints_stereo_car='/home/maximebonnesoeur/monstereo/data/arrays/joints-kitti-vehicles-stereo-201022-1536.json'
joints_mono_car='data/arrays/joints-kitti-vehicles-201028-1054.json' 

dir_ann_car='/data/maxime-data/outputs_openpifpaf_kitti/annotations_car'
dataset='kitti'

#Test apolloscape

dataset='apolloscape'
dir_ann_car='/data/maxime-data/apollo-pifpaf/annotations/'
#joints_mono_car='/home/maximebonnesoeur/monstereo/data/arrays/joints-apolloscape-train-201028-2236.json'

echo "GPU USED ${cuda_device_to_use}"

model_out () {
    while read -r line; do
        id=`echo $line | cut -c 1-12`   
        echo "$line" 

        if [ $id == "data/models/" ]    
        then     
        model=$line  
        fi 

        if [ $hyp == "1" ]
        then
        hidden_size=$line
        echo "$hidden_size"
        fi
    done <<< $1
}


joints_out () {
    while read -r line; do
        id=`echo $line | cut -c 1-18`   
        echo "$line" 
        if [ $id == "data/arrays/joints" ]
        then
        joints=$line  
        fi
    done <<< $1
}


if [ $use_car == "1" ]
    then 

    echo "CAR MODE"

    if [ $joints_there == "0" ]
    then

        echo "Command joints mono processing"
        echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --monocular --vehicles --dataset ${dataset}"

        output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --monocular --vehicles --dataset ${dataset})
        joints_out "$output"
        joints_mono_car="$joints"
    fi
    echo "Output joint file mono"
    echo "$joints_mono_car"



    echo "Command training mono"
    if [ $hyp == "1" ]
    then 
        echo "Hyper pyrameter optimization enabled with multiplier of ${multipler}"
        echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler}"
        output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler})
    else
        echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save"
        output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save)
    fi
    model_out "$output"
    model_mono="$model"
    echo "Output mono car model"
    echo "$model_mono"
    echo "$hidden_size"




    if [ $stereo == "1" ]
    then 

        if [ $joints_there == "0" ]
        then

            echo "Command joints stereo processing"
            echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --vehicles --dataset ${dataset}"

            output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --vehicles --dataset ${dataset})
            joints_out "$output"
            joints_stereo_car="$joints"
        fi
        echo "Output joint file stereo"
        echo "$joints_mono_car"



        echo "Command training stereo"
        if [ $hyp == "1" ]
        then
            echo "Hyper pyrameter optimization enabled with multiplier of ${multipler}"
            echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler}"
            output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler})
        else
            echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save"
            output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save)
        fi

        model_out "$output"
        model_stereo="$model"

        echo "output stereo car model"
        echo "$model_stereo"
    else
        model_stereo="nope"
    fi

    if [ $dataset == "kitti" ]
    then
        echo "Generate and evaluate the output"
        echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m monstereo.run eval --dir_ann ${dir_ann_car} --model ${model_stereo} --model_mono ${model_mono} --hidden_size ${hidden_size} --vehicles --generate ${eval_mode}"
        CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m monstereo.run eval --dir_ann ${dir_ann_car} --model ${model_stereo} --model_mono ${model_mono} --hidden_size ${hidden_size} --vehicles --generate ${eval_mode}
    fi

else

    echo "HUMAN MODE"


    if [ $joints_there == "0" ]
    then

        echo "Command joints mono processing"
        echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann} --monocular --dataset ${dataset}"

        output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann} --monocular --dataset ${dataset})
        joints_out "$output"
        joints_mono="$joints"
    fi
    echo "Output joint file mono"
    echo "$joints_mono"


    # train mono model
    echo "Train mono "
    output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --save)
    model_out "$output"

    model_mono="$model"

    echo "$model_mono"


    if [ $stereo == "1" ]
    then 

        if [ $joints_there == "0" ]
        then
            echo "Command joints stereo processing"
            echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann} --dataset ${dataset}"

            output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run prep --dir_ann ${dir_ann} --dataset ${dataset})
            joints_out "$output"
            joints_stereo="$joints"
        fi
        echo "Output joint file mono"
        echo "$joints_stereo"


        echo "Train stereo"
        output=$(CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run train --epochs ${epochs} --dataset ${dataset} --joints ${joints_stereo} --hidden_size ${hidden_size} --dataset kitti --save)
        model_out "$output"
        model_stereo="$model"
    else
        model_stereo="nope"
    fi

    echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m monstereo.run eval --dir_ann ${dir_ann} --model ${model_stereo}  --model_mono ${model_mono} --generate ${eval_mode}"

    CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m monstereo.run eval --dir_ann ${dir_ann} --model ${model_stereo}  --model_mono ${model_mono} --generate ${eval_mode}

fi
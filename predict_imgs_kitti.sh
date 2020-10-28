input="/home/maximebonnesoeur/monstereo/splits/kitti_train.txt"
input="/home/maximebonnesoeur/monstereo/splits/kitti_val.txt"



output_directory='visual_tests/train_kitti'
output_directory='visual_tests/val_kitti'

dir_ann_car='data/kitti-pifpaf/annotations_car'

model='data/models/ms-201027-1636-vehicles.pkl'
path_gt='data/arrays/names-kitti-vehicles-201027-1628.json'   #Careful here, this is the name folder

cuda_device_to_use='1'

#Test with the mean test (something fishy is happening)

output_directory='visual_tests/val_kitti_mean'
model='data/models/ms-201027-1840-vehicles.pkl'
path_gt='data/arrays/names-kitti-vehicles-201027-1837.json' 

while IFS= read -r line
do
    CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png --joints_folder ${dir_ann_car} --mode mono --model ${model} --output_types combined --vehicles --path_gt ${path_gt} --draw_box --z_max 60 --output-directory ${output_directory} 
    echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use}  python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png --joints_folder ${dir_ann_car} --mode mono --model ${model}  --show --output_types combined --vehicles --path_gt ${path_gt} --draw_box --z_max 60"

  echo "$line"
done < "$input"
input="/home/maximebonnesoeur/monstereo/splits/kitti_train.txt"
input="/home/maximebonnesoeur/monstereo/splits/kitti_val.txt"

mode='mono'

output_directory='visual_tests/train_kitti'
output_directory='visual_tests/val_kitti'

dir_ann_car='data/kitti-pifpaf/annotations_car'


cuda_device_to_use='1'


#Test with the mean test best results

output_directory='visual_tests/val_kitti_mean'
model='data/models/ms-201030-1711-vehicles.pkl'
path_gt='data/arrays/names-kitti-vehicles-201030-1711.json' 
i=0


output_types='combined_3d'

output_directory='visual_tests/val_kitti'


while IFS= read -r line #& [ $i -le 2 ]
do
  CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png --joints_folder ${dir_ann_car} --mode ${mode} --model ${model} --output_types ${output_types} --vehicles --path_gt ${path_gt} --draw_box --z_max 60 --output-directory ${output_directory} --show
  echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png --joints_folder ${dir_ann_car} --mode ${mode} --model ${model} --output_types ${output_types} --vehicles --path_gt ${path_gt} --draw_box --z_max 60 --output-directory ${output_directory} --show"
  #CUDA_VISIBLE_DEVICES=${cuda_device_to_use}  python3 -m openpifpaf.predict --checkpoint ~/openpifpaf_apollocar3d/outputs/shufflenetv2k16w-201009-153737-apollo.pkl.epoch300  /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png --image-output ${output_directory} 
  #CUDA_VISIBLE_DEVICES=${cuda_device_to_use}  python3 -m openpifpaf.predict --checkpoint shufflenetv2k30 /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png  --image-output ${output_directory} 

  echo "$line"
  if [[ "$i" == '100' ]]; then
    break
  fi
  ((i++))

done < "$input"
input="data/apolloscape/train/split/train-list.txt"
input="data/apolloscape/train/split/train-list.txt"

mode='mono'


output_directory='visual_tests/val_apolloscape'

dir_ann_car='data/apollo-pifpaf/annotations'


cuda_device_to_use='1'


#Test with the mean test best results

model='data/models/ms-201101-1756-vehicles_apolloscape.pkl'
path_gt='data/arrays/names-apolloscape-train-201101-1658.json'
i=0


output_types='combined_3d'

while IFS=$' \t\n\r' read -r line
do
  CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/apolloscape/train/images/$line --joints_folder ${dir_ann_car} --mode ${mode} --model ${model} --output_types ${output_types} --vehicles --path_gt ${path_gt} --draw_box --z_max 60 --output-directory ${output_directory} --show
  echo "CUDA_VISIBLE_DEVICES=${cuda_device_to_use} python3 -m  monstereo.run predict /home/maximebonnesoeur/monstereo/data/apolloscape/train/images/$line --joints_folder ${dir_ann_car} --mode ${mode} --model ${model} --output_types ${output_types} --vehicles --path_gt ${path_gt} --draw_box --z_max 60 --output-directory ${output_directory} --show"
  #CUDA_VISIBLE_DEVICES=${cuda_device_to_use}  python3 -m openpifpaf.predict --checkpoint ~/openpifpaf_apollocar3d/outputs/shufflenetv2k16w-201009-153737-apollo.pkl.epoch300  /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png --image-output ${output_directory} 
  #CUDA_VISIBLE_DEVICES=${cuda_device_to_use}  python3 -m openpifpaf.predict --checkpoint shufflenetv2k30 /home/maximebonnesoeur/monstereo/data/kitti/training/image_2/$line.png  --image-output ${output_directory} 

  echo "$line"
  if [[ "$i" == '100' ]]; then
    break
  fi
  ((i++))

done < "$input"
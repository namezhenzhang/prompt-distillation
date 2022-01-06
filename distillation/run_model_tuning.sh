# cd /mnt/sfs_turbo/zhangzhen/OpenPrompt

# mv logs/* pre_logs/*
# mkdir logs

dataset_name_list=("BoolQ" "CB" "RTE" "COPA" "MultiRC" "WiC" "WSC" ) 
for ((dataset_name_idx=1;dataset_name_idx<=1;dataset_name_idx++))
do
(
    echo "dataset name: ${dataset_name_list[${dataset_name_idx}]}"
    python distillation/main_copy.py \
    --config_yaml_model_tuning distillation/model_tuning_config.yaml \
    --auto_shorten_valid_dataset \
    --mode model_tuning \
    --dataset_name_ ${dataset_name_list[${dataset_name_idx}]} \
    --dataset_path_ datasets/FewGLUE/${dataset_name_list[${dataset_name_idx}]} \
    --template_path_ my_code/scripts/SuperGLUE/${dataset_name_list[${dataset_name_idx}]}/soft_template.txt \
    --manual_verbalizer_path_ my_code/scripts/SuperGLUE/${dataset_name_list[${dataset_name_idx}]}/manual_verbalizer.txt  
)&
done
wait
# -rm logs/**/best.ckpt
# -rm logs/**/last.ckpt
echo finish
#/super_glue.`echo ${dataset_name_list[${dataset_name_idx}]} | tr 'A-Z' 'a-z'` \ 
#--dataset_path_ datasets/FewGLUE \   
# --replace_test_by_valid \
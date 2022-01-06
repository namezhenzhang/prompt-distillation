# cd /mnt/sfs_turbo/zhangzhen/OpenPrompt
export CUDA_VISIBLE_DEVICES=4,5
dataset_name_list=("BoolQ" "CB" "COPA" "MultiRC" "RECORD" "RTE" "WiC" "WSC")

#for ((dataset_name_idx=0;dataset_name_idx<=2;dataset_name_idx++))
for myseed in  1 2 3 4 5 6
do
(
    for dataset_name_idx in 5
    do
    (
        echo "dataset name: ${dataset_name_list[${dataset_name_idx}]}"

        python my_code/exp/main.py \
        --seed $myseed \
        --use_cuda True \
        --model_parallelize True \
        --learning_setting full \
        --warmup_step_model_tuning 500 \
        --warmup_step_prompt_tuning 500 \
        --mode prompt_tuning \
        --fewshot_seeds 16 26 36 46 56 \
        --few_shot_num_examples_per_label 16 \
        --few_shot_also_sample_dev True \
        --few_shot_num_examples_per_label_dev 16 \
        --soft_token_num 100 \
        --max_seq_l 300 \
        --fine_tuning_lr 1e-5 \
        --model_name t5 \
        --model_path t5-large \
        --optimizer adamw \
        --train_batch_size 8 \
        --eval_batch_size 64 \
        --tuning_max_steps 3000 \
        --distill_max_steps 15000 \
        --distill_training_total_steps 20000 \
        --gradient_accumulation_steps 1 \
        --tuning_eval_every_steps 8 \
        --distill_eval_every_steps 20 \
        --temperature 8 \
        --alpha 0.3 \
        --beta 0.3 \
        --gamma 0.2 \
        --save_ckpt False \
        --dataset_name ${dataset_name_list[${dataset_name_idx}]} \
        --dataset_path datasets/FewGLUE/${dataset_name_list[${dataset_name_idx}]}/ \
        --template_path my_code/scripts/SuperGLUE/${dataset_name_list[${dataset_name_idx}]}/soft_template_UTF.txt \
        --manual_verbalizer_path my_code/scripts/SuperGLUE/${dataset_name_list[${dataset_name_idx}]}/manual_verbalizer_UTF.txt \
    )
    done
)
done
# wait
echo finish



# cd /mnt/sfs_turbo/zhangzhen/OpenPrompt
# export CUDA_VISIBLE_DEVICES=2,7
# dataset_name_list=("BoolQ" "CB" "COPA" "MultiRC" "RECORD" "RTE" "WiC" "WSC")

# #for ((dataset_name_idx=0;dataset_name_idx<=2;dataset_name_idx++))
# for dataset_name_idx in 0 1 5
# do
# (
#     echo "dataset name: ${dataset_name_list[${dataset_name_idx}]}"

#     python my_code/exp/main.py \
#     --seed 100 \
#     --use_cuda True \
#     --model_parallelize True \
#     --learning_setting few_shot \
#     --mode prompt_tuning \
#     --fewshot_seeds 16 26 36 46 56 \
#     --few_shot_num_examples_per_label 16 \
#     --few_shot_also_sample_dev True \
#     --few_shot_num_examples_per_label_dev 16 \
#     --soft_token_num 20 \
#     --model_name t5 \
#     --model_path t5-large \
#     --train_batch_size 4 \
#     --eval_batch_size 64 \
#     --tuning_max_steps 20000 \
#     --distill_max_steps 15000 \
#     --distill_training_total_steps 25000 \
#     --gradient_accumulation_steps 8 \
#     --tuning_eval_every_steps 1000 \
#     --distill_eval_every_steps 1000 \
#     --temperature 8 \
#     --alpha 0.3 \
#     --beta 0.3 \
#     --gamma 0.2 \
#     --save_ckpt False \
#     --dataset_name super_glue.${dataset_name_list[${dataset_name_idx}]} \
#     --template_path my_code/scripts/SuperGLUE/${dataset_name_list[${dataset_name_idx}]}/soft_template.txt \
#     --manual_verbalizer_path my_code/scripts/SuperGLUE/${dataset_name_list[${dataset_name_idx}]}/manual_verbalizer.txt \
# )
# done
# # wait
# echo finish
#    --replace_test_by_valid \

# --dataset_path /mnt/sfs_turbo/huggingface_datasets/super_glue.`echo ${dataset_name_list[${dataset_name_idx}]} | tr 'A-Z' 'a-z'` \
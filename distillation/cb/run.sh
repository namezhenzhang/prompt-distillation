
python distillation/main_.py \
--config_yaml_distillation distillation/cb/distillation_config.yaml \
--config_yaml_model_tuning distillation/cb/model_tuning_config.yaml \
--resume_model_tuning logs/cb_model_tuning_teacher_model/seed-123 \
--replace_test_by_valid \
# > distillation/cb/output.txt 2>&1




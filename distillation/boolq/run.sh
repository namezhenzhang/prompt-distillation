python distillation/main_.py \
--config_yaml_distillation distillation/prompt_tuning_config.yaml \
--config_yaml_model_tuning distillation/model_tuning_config.yaml \
--replace_test_by_valid \
> distillation/output.txt 2>&1



#--resume_model_tuning logs/super_glue.boolq_t5-large_manual_template_manual_verbalizer_1125204157822654/seed-123 \
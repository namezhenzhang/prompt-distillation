import os
import sys
sys.path.append(".")

import argparse

from openprompt.trainer import ClassificationRunner, GenerationRunner
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from re import template
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.prompts import load_template, load_verbalizer, load_template_generator, load_verbalizer_generator
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.config import get_config, save_config_to_yaml
from openprompt.plms import load_plm_from_config
from openprompt.data_utils import load_dataset
from openprompt.utils.cuda import model_to_device

from distillation_utils import *



def trainer(EXP_PATH, config, main_config, Processor, train_dataset = None, valid_dataset = None, test_dataset = None, resume_model_tuning=None, mode=None, resume = None, test = None, zero = False):
    if mode == 'distillation':
        distillation_config = main_config
        model_or_prompt_tuning_config = config
    elif mode == 'prompt_tuning' or mode == 'model_tuning':
        distillation_config = None
        model_or_prompt_tuning_config = main_config

    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)

    if distillation_config!=None:
        distillation_config.logging.path = EXP_PATH
    model_or_prompt_tuning_config.logging.path = EXP_PATH

    # set seed
    set_seed(main_config.reproduce.seed)

    # load the pretrained models, its model, tokenizer, and config.
    model_or_prompt_tuning_plm_model, model_or_prompt_tuning_plm_tokenizer, model_or_prompt_tuning_plm_config, model_or_prompt_tuning_plm_wrapper_class = load_plm_from_config(model_or_prompt_tuning_config)

    # define template and verbalizer
    if model_or_prompt_tuning_config.task == "classification":
        # model tuning
        model_or_prompt_tuning_template = load_template(config=model_or_prompt_tuning_config, model=model_or_prompt_tuning_plm_model, tokenizer=model_or_prompt_tuning_plm_tokenizer, plm_config=model_or_prompt_tuning_plm_config)
        model_or_prompt_tuning_verbalizer = load_verbalizer(config=model_or_prompt_tuning_config, model=model_or_prompt_tuning_plm_model, tokenizer=model_or_prompt_tuning_plm_tokenizer, plm_config=model_or_prompt_tuning_plm_config, classes=Processor.labels)
        model_or_prompt_tuning_prompt_model = PromptForClassification(model_or_prompt_tuning_plm_model, model_or_prompt_tuning_template, model_or_prompt_tuning_verbalizer, freeze_plm = model_or_prompt_tuning_config.plm.optimize.freeze_para)

            
    elif model_or_prompt_tuning_config.task == "generation":
        raise NotImplementedError('have not implement generation')
        # template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        # prompt_model = PromptForGeneration(plm_model, template, freeze_plm = config.plm.optimize.freeze_para, gen_config = config.generation)
    else:
        raise NotImplementedError(f"config.task {distillation_config.task} is not implemented yet. Only classification and generation are supported.")

    # process data and get data_loader
    train_dataloader = build_dataloader(train_dataset, model_or_prompt_tuning_template, model_or_prompt_tuning_plm_tokenizer, model_or_prompt_tuning_plm_wrapper_class, model_or_prompt_tuning_config, "train") if train_dataset else None
    valid_dataloader = build_dataloader(valid_dataset, model_or_prompt_tuning_template, model_or_prompt_tuning_plm_tokenizer, model_or_prompt_tuning_plm_wrapper_class, model_or_prompt_tuning_config, "dev") if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, model_or_prompt_tuning_template, model_or_prompt_tuning_plm_tokenizer, model_or_prompt_tuning_plm_wrapper_class, model_or_prompt_tuning_config, "test") if test_dataset else None

    if model_or_prompt_tuning_config.task == "classification":
        if model_or_prompt_tuning_config.classification.auto_t or model_or_prompt_tuning_config.classification.auto_v:
            raise NotImplementedError('have not implement auto classification')
            runner = LMBFFClassificationRunner(train_dataset = train_dataset, 
                                                valid_dataset = valid_dataset, 
                                                test_dataset = test_dataset, 
                                                template=template,
                                                verbalizer=distillation_verbalizer,
                                                config = distillation_config
                                                )
        else:
            model_or_prompt_tuning_runner = ClassificationRunner(model = model_or_prompt_tuning_prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    id2label = Processor.id2label,
                                    config = model_or_prompt_tuning_config
            )
            if resume_model_tuning:
                logger.info(f'Loading model tuning model from {resume_model_tuning}')
                model_or_prompt_tuning_config.logging.path ,resume_model_tuning = resume_model_tuning, model_or_prompt_tuning_config.logging.path 
            else:
                logger.info("Begin model tuning.")
                res = model_or_prompt_tuning_runner.run()


    elif model_or_prompt_tuning_config.task == "generation":
        raise NotImplementedError('have not implement generation')
        runner = GenerationRunner(
            model = prompt_model,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            test_dataloader = test_dataloader,
            config = distillation_config
        )

    if mode == 'distillation':
        distillation_plm_model, distillation_plm_tokenizer, distillation_plm_config, distillation_plm_wrapper_class = load_plm_from_config(distillation_config)

        # define template and verbalizer
        if distillation_config.task == "classification":
            distillation_template = load_template(config=distillation_config, model=distillation_plm_model, tokenizer=distillation_plm_tokenizer, plm_config=distillation_plm_config)
            distillation_verbalizer = load_verbalizer(config=distillation_config, model=distillation_plm_model, tokenizer=distillation_plm_tokenizer, plm_config=distillation_plm_config, classes=Processor.labels)
            distillation_prompt_model = PromptForClassification(distillation_plm_model, distillation_template, distillation_verbalizer, freeze_plm = distillation_config.plm.optimize.freeze_para)

        # runner
        if distillation_config.task == "classification":
            model_or_prompt_tuning_runner.load_checkpoint('best', False)
            if resume_model_tuning:
                model_or_prompt_tuning_config.logging.path ,resume_model_tuning = resume_model_tuning, model_or_prompt_tuning_config.logging.path
            model_or_prompt_tuning_prompt_model.eval()
            distillation_runner = Distillation_classification_Runner(prompt_tuning_model=distillation_prompt_model,
                                                                model_tuning_model=model_or_prompt_tuning_prompt_model,
                                                                config=distillation_config,
                                                                model_tuning_config=model_or_prompt_tuning_config,
                                                                train_dataloader=train_dataloader,
                                                                valid_dataloader=valid_dataloader,
                                                                test_dataloader=test_dataloader,
                                                                id2label=Processor.id2label,
                                                                loss_function=distribution_loss
                                                                )

            logger.info("Begin distillation.")
            res = distillation_runner.run()

    return res


def change_template_path(config, args):
    
    if args.dataset_name_ != 'None':
        config.dataset.name = args.dataset_name_
    if args.dataset_path_ != 'None':
        config.dataset.path = args.dataset_path_
    if args.template_path_ != 'None':
        try:
            config.soft_template.file_path = args.template_path_
        except:
            config.manual_template.file_path = args.template_path_
    if args.manual_verbalizer_path_ != 'None':
        config.manual_verbalizer.file_path = args.manual_verbalizer_path_





def main():
    args = get_my_args()
    main_config, config = get_my_config(args,args.mode)

    change_template_path(main_config, args)
    if config != None:
        change_template_path(config, args)

    logger.info(f'dataset name: {main_config.dataset.name}')
    logger.info(f'dataset path: {main_config.dataset.path}')
    try:
        logger.info(f'template path: {main_config.soft_template.file_path}')
    except:
        logger.info(f'template path: {main_config.manual_template.file_path}')
    logger.info(f'verbalizer path: {main_config.manual_verbalizer.file_path}')

    EXP_PATH = config_experiment_dir(main_config)



    init_logger(os.path.join(EXP_PATH, "log.txt"), main_config.logging.file_level, main_config.logging.console_level)
    # save config to the logger directory
    save_config_to_yaml(main_config)

    logger.info(f'dataset name: {main_config.dataset.name}')
    logger.info(f'dataset path: {main_config.dataset.path}')
    try:
        logger.info(f'template path: {main_config.soft_template.file_path}')
    except:
        logger.info(f'template path: {main_config.manual_template.file_path}')
    logger.info(f'verbalizer path: {main_config.manual_verbalizer.file_path}')
    

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(main_config, test = args.test is not None or main_config.learning_setting == 'zero_shot')
    
    if args.replace_test_by_valid:
        logger.info('replace test by valid')
        test_dataset = valid_dataset
        
    # main
    if main_config.learning_setting == 'full':
        logger.info(f'dataset len: train:{len(train_dataset)}, valid:{len(valid_dataset)}, test:{len(test_dataset)}')
        res = trainer(
            EXP_PATH,
            config,
            main_config,
            Processor,
            resume = args.resume,
            resume_model_tuning=args.resume_model_tuning,
            mode=args.mode,
            test = args.test,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            test_dataset = test_dataset,
        )
    elif main_config.learning_setting == 'few_shot':
        if main_config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = main_config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label = main_config.sampling_from_train.num_examples_per_label,
                    also_sample_dev = main_config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev = main_config.sampling_from_train.num_examples_per_label_dev
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset = train_dataset,
                    valid_dataset = valid_dataset,
                    seed = seed
                )
                
                valid_dataset_used = None
                if args.auto_shorten_valid_dataset:
                    if len(valid_dataset) > 300:
                        logger.info('auto shorten')
                        valid_dataset_used = valid_sampled_dataset
                    else:
                        valid_dataset_used = valid_dataset
                else:
                    valid_dataset_used = valid_sampled_dataset

                logger.info(f'dataset len: train:{len(train_sampled_dataset)}, valid:{len(valid_dataset_used)}, test:{len(test_dataset)}')

                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    main_config,
                    Processor,
                    resume = args.resume,
                    resume_model_tuning=args.resume_model_tuning,
                    mode=args.mode,
                    test = args.test,
                    train_dataset = train_sampled_dataset,
                    valid_dataset = valid_dataset_used,
                    test_dataset = test_dataset,
                )
            else:
                raise NotImplementedError('have not implement generation')
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    model_tuning_config,
                    distillation_config,
                    Processor,
                    resume_model_tuning=args.resume_model_tuning,
                    test = args.test,
                    test_dataset = test_dataset,
                )
            res += result
        res /= len(seeds)
    elif main_config.learning_setting == 'zero_shot':
        raise NotImplementedError('have not implement generation')
        res = trainer(
            EXP_PATH,
            model_tuning_config,
            distillation_config,
            Processor,
            zero = True,
            resume_model_tuning=args.resume_model_tuning,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            test_dataset = test_dataset,
        )

    print("metric:", res)

if __name__ == "__main__":
    # sys.argv = ['distillation/main_.py', \
    #     '--config_yaml_distillation', 'distillation/prompt_tuning_config.yaml',     
    #     '--config_yaml_model_tuning', 'distillation/model_tuning_config.yaml', 
    #     '--resume_model_tuning', 'logs/super_glue.boolq_t5-large_manual_template_manual_verbalizer_1125204157822654/seed-123', 
    #     '--replace_test_by_valid']
    main()
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



def trainer(EXP_PATH, model_tuning_config, distillation_config, Processor, train_dataset = None, valid_dataset = None, test_dataset = None, resume_model_tuning=None, resume = None, test = None, zero = False):
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    distillation_config.logging.path = EXP_PATH
    model_tuning_config.logging.path = EXP_PATH
    # set seed
    set_seed(distillation_config.reproduce.seed)

    # load the pretrained models, its model, tokenizer, and config.
    model_tuning_plm_model, model_tuning_plm_tokenizer, model_tuning_plm_config, model_tuning_plm_wrapper_class = load_plm_from_config(model_tuning_config)

    distillation_plm_model, distillation_plm_tokenizer, distillation_plm_config, distillation_plm_wrapper_class = load_plm_from_config(distillation_config)

    

    # define template and verbalizer
    if distillation_config.task == "classification":
        # model tuning
        model_tuning_template = load_template(config=model_tuning_config, model=model_tuning_plm_model, tokenizer=model_tuning_plm_tokenizer, plm_config=model_tuning_plm_config)
        model_tuning_verbalizer = load_verbalizer(config=model_tuning_config, model=model_tuning_plm_model, tokenizer=model_tuning_plm_tokenizer, plm_config=model_tuning_plm_config, classes=Processor.labels)
        model_tuning_prompt_model = PromptForClassification(model_tuning_plm_model, model_tuning_template, model_tuning_verbalizer, freeze_plm = model_tuning_config.plm.optimize.freeze_para)

        # distillation 
        distillation_template = load_template(config=distillation_config, model=distillation_plm_model, tokenizer=distillation_plm_tokenizer, plm_config=distillation_plm_config)
        distillation_verbalizer = load_verbalizer(config=distillation_config, model=distillation_plm_model, tokenizer=distillation_plm_tokenizer, plm_config=distillation_plm_config, classes=Processor.labels)
        distillation_prompt_model = PromptForClassification(distillation_plm_model, distillation_template, distillation_verbalizer, freeze_plm = distillation_config.plm.optimize.freeze_para)
            
    elif distillation_config.task == "generation":
        raise NotImplementedError('have not implement generation')
        # template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        # prompt_model = PromptForGeneration(plm_model, template, freeze_plm = config.plm.optimize.freeze_para, gen_config = config.generation)
    else:
        raise NotImplementedError(f"config.task {distillation_config.task} is not implemented yet. Only classification and generation are supported.")

    # process data and get data_loader
    train_dataloader = build_dataloader(train_dataset, distillation_template, distillation_plm_tokenizer, distillation_plm_wrapper_class, distillation_config, "train") if train_dataset else None
    valid_dataloader = build_dataloader(valid_dataset, distillation_template, distillation_plm_tokenizer, distillation_plm_wrapper_class, distillation_config, "dev") if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, distillation_template, distillation_plm_tokenizer, distillation_plm_wrapper_class, distillation_config, "test") if test_dataset else None

    if distillation_config.task == "classification":
        if distillation_config.classification.auto_t or distillation_config.classification.auto_v:
            raise NotImplementedError('have not implement auto classification')
            exit(1)
            runner = LMBFFClassificationRunner(train_dataset = train_dataset, 
                                                valid_dataset = valid_dataset, 
                                                test_dataset = test_dataset, 
                                                template=template,
                                                verbalizer=distillation_verbalizer,
                                                config = distillation_config
                                                )
        else:
            model_tuning_runner = ClassificationRunner(model = model_tuning_prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    id2label = Processor.id2label,
                                    config = model_tuning_config
            )
            if resume_model_tuning:
                logger.info(f'Loading model tuning model from {resume_model_tuning}')
                model_tuning_config.logging.path ,resume_model_tuning = resume_model_tuning, model_tuning_config.logging.path 
                model_tuning_runner.load_checkpoint('best', False)
                model_tuning_config.logging.path ,resume_model_tuning = resume_model_tuning, model_tuning_config.logging.path 
            else:
                logger.info("Begin model tuning.")
                model_tuning_runner.run()
                model_tuning_runner.load_checkpoint('best', False)
            distillation_runner = Distillation_classification_Runner(prompt_tuning_model=distillation_prompt_model,
                                                            model_tuning_model=model_tuning_prompt_model,
                                                            config=distillation_config,
                                                            model_tuning_config=model_tuning_config,
                                                            train_dataloader=train_dataloader,
                                                            valid_dataloader=valid_dataloader,
                                                            test_dataloader=test_dataloader,
                                                            id2label=Processor.id2label,
                                                            loss_function=distribution_loss
                                                            )
    elif distillation_config.task == "generation":
        raise NotImplementedError('have not implement generation')
        

        # runner = GenerationRunner(
        #     model = prompt_model,
        #     train_dataloader = train_dataloader,
        #     valid_dataloader = valid_dataloader,
        #     test_dataloader = test_dataloader,
        #     config = distillation_config
        # )
        
    # if resume_model_tuning:
    #     logger.info(f'Loading model tuning model from {resume_model_tuning}')
    #     model_tuning_config.logging.path ,resume_model_tuning = resume_model_tuning, model_tuning_config.logging.path 
    #     model_tuning_runner.load_checkpoint('best', False)
    #     model_tuning_config.logging.path ,resume_model_tuning = resume_model_tuning, model_tuning_config.logging.path 
    # else:
    #     logger.info("Begin model tuning.")
    #     model_tuning_runner.run()

    model_tuning_prompt_model.eval()

    logger.info("Begin distillation.")
    res = distillation_runner.run()



    # if zero:
    #     res = runner.test()
    # elif test:
    #     res = runner.test(ckpt = 'best')
    # elif resume:
    #     res = runner.run(ckpt = 'last')
    # else:
    #     res = runner.run()


    return res


def change_template_path(distillation_config, model_tuning_config, args):
    
    if args.dataset_name_ != 'None':
        distillation_config.dataset.name = args.dataset_name_
        model_tuning_config.dataset.name = args.dataset_name_
    logger.info(f'dataset name: {distillation_config.dataset.name}')
    if args.dataset_path_ != 'None':
        distillation_config.dataset.path = args.dataset_path_
        model_tuning_config.dataset.path = args.dataset_path_
    if args.template_path_ != 'None':
        distillation_config.soft_template.file_path = args.template_path_
        model_tuning_config.manual_template.file_path = args.template_path_
    logger.info(f'template path: {distillation_config.soft_template.file_path}')

    if args.manual_verbalizer_path_ != 'None':
        distillation_config.manual_verbalizer.file_path = args.manual_verbalizer_path_
        model_tuning_config.manual_verbalizer.file_path = args.manual_verbalizer_path_
    logger.info(f'verbalizer path: {distillation_config.manual_verbalizer.file_path}')





def main():

    distillation_config, model_tuning_config, args = get_my_config()

    change_template_path(distillation_config, model_tuning_config, args)

    EXP_PATH = config_experiment_dir(distillation_config)
    init_logger(os.path.join(EXP_PATH, "log.txt"), distillation_config.logging.file_level, distillation_config.logging.console_level)
    # save config to the logger directory
    save_config_to_yaml(distillation_config)

    logger.info(f'dataset name: {distillation_config.dataset.name}')
    logger.info(f'dataset path: {distillation_config.dataset.path}')
    try:
        logger.info(f'template path: {distillation_config.soft_template.file_path}')
    except:
        logger.info(f'template path: {distillation_config.manual_template.file_path}')
    logger.info(f'verbalizer path: {distillation_config.manual_verbalizer.file_path}')
    

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(distillation_config, test = args.test is not None or distillation_config.learning_setting == 'zero_shot')
    
    if args.replace_test_by_valid:
        test_dataset = valid_dataset
        
    # main
    if distillation_config.learning_setting == 'full':
        logger.info(f'dataset len: train:{len(train_dataset)}, valid:{len(valid_dataset)}, test:{len(test_dataset)}')
        res = trainer(
            EXP_PATH,
            model_tuning_config,
            distillation_config,
            Processor,
            resume = args.resume,
            resume_model_tuning=args.resume_model_tuning,
            test = args.test,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            test_dataset = test_dataset,
        )
    elif distillation_config.learning_setting == 'few_shot':
        if distillation_config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = distillation_config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label = distillation_config.sampling_from_train.num_examples_per_label,
                    also_sample_dev = distillation_config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev = distillation_config.sampling_from_train.num_examples_per_label_dev
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset = train_dataset,
                    valid_dataset = valid_dataset,
                    seed = seed
                )
                
                valid_dataset_used = None
                if args.auto_shorten_valid_dataset:
                    if len(valid_dataset) > 300:
                        valid_dataset_used = valid_sampled_dataset
                    else:
                        valid_dataset_used = valid_dataset
                else:
                    valid_dataset_used = valid_sampled_dataset

                logger.info(f'dataset len: train:{len(train_sampled_dataset)}, valid:{len(valid_dataset_used)}, test:{len(test_dataset)}')

                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    model_tuning_config,
                    distillation_config,
                    Processor,
                    resume = args.resume,
                    resume_model_tuning=args.resume_model_tuning,
                    test = args.test,
                    train_dataset = train_sampled_dataset,
                    valid_dataset = valid_dataset_used,
                    test_dataset = test_dataset,
                )
            else:
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
    elif distillation_config.learning_setting == 'zero_shot':
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
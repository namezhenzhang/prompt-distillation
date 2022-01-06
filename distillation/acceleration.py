import sys
sys.path.append(".")
from yacs.config import CfgNode
from torch._C import device
from openprompt.utils.reproduciblity import set_seed
from openprompt.utils.metrics import classification_metrics, generation_metric
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.utils.cuda import model_to_device
from openprompt.trainer import (BaseRunner, ClassificationRunner,
                                GenerationRunner)
from openprompt.prompts import (SoftTemplate, load_template,
                                load_template_generator, load_verbalizer,
                                load_verbalizer_generator)
from openprompt.plms import get_model_class, load_plm_from_config
from openprompt.pipeline_base import (PromptForClassification,
                                      PromptForGeneration)
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from openprompt.data_utils import FewShotSampler, load_dataset
from openprompt.config import (check_config_conflicts, get_config,
                               get_user_config, save_config_to_yaml)
from openprompt import PromptDataLoader
import torch.nn.functional as F
import torch
from typing import Callable, Dict, Optional, Union
from re import template
import time
import argparse
import os
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict


from distillation.distillation_utils import Prompt_acceleration_Runner, distribution_loss,get_my_config,build_dataloader



def trainer(EXP_PATH, config, Processor, train_dataset=None, valid_dataset=None, test_dataset=None, resume_model_tuning=None, test=None, zero=False):
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    config, model_tuning_config = config
    config.logging.path = EXP_PATH
    model_tuning_config.logging.path = EXP_PATH
    # set seed
    set_seed(config.reproduce.seed)


    #---------------------------------------------------------------------------------------------------
    # load the pretrained models, its model, tokenizer, and config.
    plm_config = config.plm
    model_class = get_model_class(plm_type=plm_config.model_name)
    model_tuning_model_config = model_class.config.from_pretrained(plm_config.model_path)
    prompt_tuning_model_config = model_class.config.from_pretrained(plm_config.model_path)

    model_tuning_model = model_class.model.from_pretrained(
        plm_config.model_path, config=model_tuning_model_config)
    prompt_tuning_model = model_class.model.from_pretrained(
        plm_config.model_path, config=prompt_tuning_model_config)

    plm_tokenizer = model_class.tokenizer.from_pretrained(
        plm_config.model_path)
    plm_wrapper_class = model_class.wrapper


    # define template and verbalizer
    model_tuning_template = load_template(
        config=model_tuning_config, model=model_tuning_model, tokenizer=plm_tokenizer, plm_config=plm_config)
    prompt_tuning_template = load_template(
        config=config, model=prompt_tuning_model, tokenizer=plm_tokenizer, plm_config=plm_config)

    model_tuning_verbalizer = load_verbalizer(
        config=model_tuning_config, model=model_tuning_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
    prompt_tuning_verbalizer = load_verbalizer(
        config=config, model=prompt_tuning_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)

    # load prompt’s pipeline model
    model_tuning_prompt_model = PromptForClassification(
        model_tuning_model, model_tuning_template, model_tuning_verbalizer, freeze_plm=model_tuning_config.plm.optimize.freeze_para)

    prompt_tuning_prompt_model = PromptForClassification(
        prompt_tuning_model, prompt_tuning_template, prompt_tuning_verbalizer, freeze_plm=config.plm.optimize.freeze_para)

    #--------------------------------------------------------------------------------



    # process data and get data_loader
    train_dataloader = valid_dataloader = test_dataloader = None
    train_dataloader = build_dataloader(
        train_dataset, model_tuning_template, plm_tokenizer, plm_wrapper_class, config, "train") if train_dataset else None
    valid_dataloader = build_dataloader(
        valid_dataset, model_tuning_template, plm_tokenizer, plm_wrapper_class, config, "dev") if valid_dataset else None
    # test_dataloader = valid_dataloader
    test_dataloader = build_dataloader(
        test_dataset, model_tuning_template, plm_tokenizer, plm_wrapper_class, config, "test") if test_dataset else None

    # define modeltuning runner
    model_tuning_runner = ClassificationRunner(model=model_tuning_prompt_model,
                                               train_dataloader=train_dataloader,
                                               valid_dataloader=valid_dataloader,
                                               test_dataloader=test_dataloader,
                                               id2label=Processor.id2label,
                                               config=model_tuning_config
                                               )
    if resume_model_tuning:
        logger.info(f'Loading model tuning model from {resume_model_tuning}')
        model_tuning_config.logging.path = resume_model_tuning
        model_tuning_runner.load_checkpoint('best', False)
    else:
        logger.info("Begin model tuning.")
        res = model_tuning_runner.run()


    # define distillation runner
    # config.train.num_training_steps, config.train.acceleration_num_training_steps = config.train.acceleration_num_training_steps, config.train.num_training_steps
    # config.soft_template.optimize.lr, config.soft_template.optimize.acceleration_lr = config.soft_template.optimize.acceleration_lr, config.soft_template.optimize.lr
    model_tuning_prompt_model.eval()
    prompt_acceleration_Runner = Prompt_acceleration_Runner(prompt_tuning_model=prompt_tuning_prompt_model,
                                                            model_tuning_model=model_tuning_prompt_model,
                                                            config=config,
                                                            model_tuning_config=model_tuning_config,
                                                            train_dataloader=train_dataloader,
                                                            valid_dataloader=valid_dataloader,
                                                            test_dataloader=test_dataloader,
                                                            id2label=Processor.id2label,
                                                            loss_function=distribution_loss
                                                            )
    # last_model = 'logs/super_glue.boolq_t5-large_soft_template_manual_verbalizer_1121143957780203'
    # logger.info(f'Loading prompt tuning last model from {last_model}')
    # a = config.logging.path
    # config.logging.path = last_model
    # prompt_acceleration_Runner.load_checkpoint('last', False)
    # config.logging.path = a
    logger.info("Begin acceleration.")
    res = prompt_acceleration_Runner.run()
    # config.train.num_training_steps, config.train.acceleration_num_training_steps = config.train.acceleration_num_training_steps, config.train.num_training_steps
    # config.soft_template.optimize.lr, config.soft_template.optimize.acceleration_lr = config.soft_template.optimize.acceleration_lr, config.soft_template.optimize.lr


    # 观察modeltuning是否被改变了
    # logger.info('观察model是否被改变了')
    # model_tuning_runner.inference_epoch("validation")

    # define prompttuning runner
    # prompt_tuning_runner = ClassificationRunner(model=prompt_tuning_prompt_model,
    #                                             train_dataloader=train_dataloader,
    #                                             valid_dataloader=valid_dataloader,
    #                                             test_dataloader=test_dataloader,
    #                                             id2label=Processor.id2label,
    #                                             config=config
    #                                             )
    # logger.info("Begin prompt tuning.")
    # prompt_tuning_runner.load_checkpoint('best', False)
    # res = prompt_tuning_runner.run()
    return res

def main():
    torch.manual_seed(int(time.time()))
    config, model_tuning_config, args = get_my_config()

    EXP_PATH = config_experiment_dir(config)
    init_logger(os.path.join(EXP_PATH, "log.txt"),
                config.logging.file_level, config.logging.console_level)

    logger.info(
        f'acceleration_lr: {config.soft_template.optimize.acceleration_lr}')
    logger.info(f'dataset: {config.dataset.name}')

    # save config to the logger directory
    save_config_to_yaml(config)

    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(
        config, test=args.test is not None or config.learning_setting == 'zero_shot')

    res = trainer(
        EXP_PATH,
        (config, model_tuning_config),
        Processor,
        resume_model_tuning=args.resume_model_tuning,
        test=args.test,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    )


if __name__ == "__main__":
    # sys.argv = ["experiments/acceleration.py", "--config_yaml1", "experiments/prompt_tuning_config.yaml", "--config_yaml2", "experiments/model_tuning_config.yaml"]
    main()

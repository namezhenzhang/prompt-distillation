import argparse
import logging
import os
import time
import sys
sys.path.append(".")
import numpy as np
import torch
from tensorboardX import SummaryWriter
from openprompt import PromptDataLoader
from openprompt.data_utils import FewShotSampler
from openprompt.pipeline_base import PromptForGeneration
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, ManualTemplate, ManualVerbalizer
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.utils.reproduciblity import set_seed
from tqdm import tqdm
# use AdamW is a standard practice for transformer
from transformers import (AdamW, get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
# use Adafactor is the default setting for T5
from transformers.optimization import Adafactor, AdafactorSchedule

from utils import (Distiller, get_experiment_dir,
                   get_my_args, load_dataset, save_args)
DECODER_MAX_LENGTH_DICT = {
    'super_glue.cb': 10,
    'super_glue.copa': 50,
    'super_glue.multirc': 10,
    'super_glue.record': 20,
    'super_glue.rte': 10,
    'super_glue.wic': 10,
    'super_glue.wsc': 10,
    'super_glue.boolq': 10,

    'cb': 10,
    'copa': 50,
    'multirc': 10,
    'record': 20,
    'rte': 10,
    'wic': 10,
    'wsc': 10,
    'boolq': 10
}
METRICS = {
    'glue-cola': 'Matthew-Correlation',
    'glue-mnli': 'ACC',
    'glue-mrpc': 'ACC',
    'glue-qnli': 'ACC',
    'glue-qqp': 'ACC',
    'glue-rte': 'ACC',
    'glue-sst2': 'ACC',
    'glue-wnli': 'ACC',

    'super_glue.cb': 'Classification-F1',
    'super_glue.copa': 'ACC',
    'super_glue.multirc': 'EM',
    'super_glue.record': 'QA-F1',
    'super_glue.rte': 'ACC',
    'super_glue.wic': 'ACC',
    'super_glue.wsc': 'ACC',
    'super_glue.boolq': 'ACC',

    'cb': 'Classification-F1',
    'copa': 'ACC',
    'multirc': 'EM',
    'record': 'QA-F1',
    'rte': 'ACC',
    'wic': 'ACC',
    'wsc': 'ACC',
    'boolq': 'ACC',
}


def evaluate(args, prompt_model, dataloader,print_=False):
    generation_arguments = {
        "max_length": DECODER_MAX_LENGTH_DICT[(args.dataset_name).lower()],
    }
    predictions = []
    ground_truths = []
    # logger.info(f"eval dataset len: {len(dataloader.dataloader.dataset)}")
    for step, inputs in enumerate(dataloader):
        if args.use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(
            inputs, **generation_arguments, verbose=False)
        predictions.extend(output_sentence)
        ground_truths.extend(inputs['tgt_text'])
    assert len(predictions) == len(
        ground_truths), (len(predictions), len(ground_truths))
    predictions = [prediction.strip() for prediction in predictions]
    ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
    # shown one example
    logger.info(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
    # logger.info(f"predictions {predictions[-1]}, ground_truths {ground_truths[-1]}")
    if print_==True:
        logger.info(predictions)
        logger.info(ground_truths)
    
    score = crossfit_evaluate(
        predictions, ground_truths, metric=METRICS[(args.dataset_name).lower()])
    return score



def trainer(EXP_PATH, args, Processor, train_dataset=None, valid_dataset=None, test_dataset=None):
    #================================================================================#
    # 预处理
    logger.info("begain tuning")
    writer = SummaryWriter(os.path.join(EXP_PATH, 'tuning_tensorboard'))
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    tuning_model_save_path = os.path.join(EXP_PATH, 'tuning_ckpt')
    if not os.path.exists(tuning_model_save_path):
        os.mkdir(tuning_model_save_path)

    set_seed(args.seed)
    #================================================================================#
    # load plm，verbalizer，prompt model
    model_or_prompt_tuning_plm_model, model_or_prompt_tuning_plm_tokenizer, model_or_prompt_tuning_plm_config, model_or_prompt_tuning_plm_wrapper_class = load_plm(
        args.model_name, args.model_path)
    if args.use_cuda:
        model_or_prompt_tuning_plm_model = model_or_prompt_tuning_plm_model.cuda()

    if args.model_parallelize:
        model_or_prompt_tuning_plm_model.parallelize()

    if args.mode == 'prompt_tuning':
        model_or_prompt_tuning_template = SoftTemplate(model=model_or_prompt_tuning_plm_model, tokenizer=model_or_prompt_tuning_plm_tokenizer,
                                                       num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(args.template_path, choice=args.template_id)
        freeze_plm = True
    elif args.mode == 'fine_tuning' or args.mode == 'distillation':
        model_or_prompt_tuning_template = ManualTemplate(
            tokenizer=model_or_prompt_tuning_plm_tokenizer).from_file(args.template_path, choice=args.template_id)
        freeze_plm = False
    else:
        raise NotImplementedError()
    # model_or_prompt_tuning_verbalizer = ManualVerbalizer(model_or_prompt_tuning_plm_tokenizer,classes=Processor.get_labels()).from_file(args.manual_verbalizer_path, choice=args.manual_verbalizer_id)
    model_or_prompt_tuning_verbalizer = GenerationVerbalizer(model_or_prompt_tuning_plm_tokenizer, classes=Processor.get_labels(
    ), is_rule=False).from_file(args.manual_verbalizer_path, choice=args.manual_verbalizer_id)
    model_or_prompt_tuning_prompt_model = PromptForGeneration(
        model_or_prompt_tuning_plm_model, model_or_prompt_tuning_template, freeze_plm=freeze_plm)
    #================================================================================#
    # process data and get data_loader
    
    train_dataloader = PromptDataLoader(dataset=train_dataset, template=model_or_prompt_tuning_template, verbalizer=model_or_prompt_tuning_verbalizer, tokenizer=model_or_prompt_tuning_plm_tokenizer,  # be sure to add verbalizer
                                        # be sure to use larger decoder_max_length for teacher forcing.
                                        tokenizer_wrapper_class=model_or_prompt_tuning_plm_wrapper_class, max_seq_length=args.max_seq_l, decoder_max_length=DECODER_MAX_LENGTH_DICT[
                                            (args.dataset_name).lower()],
                                        # be sure to use teacher_forcing and predict_eos_token=True
                                        batch_size=args.train_batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                        truncate_method="tail") if train_dataset else None

    valid_dataloader = PromptDataLoader(dataset=valid_dataset, template=model_or_prompt_tuning_template, verbalizer=model_or_prompt_tuning_verbalizer, tokenizer=model_or_prompt_tuning_plm_tokenizer,
                                        tokenizer_wrapper_class=model_or_prompt_tuning_plm_wrapper_class, max_seq_length=args.max_seq_l, decoder_max_length=3,
                                        # predict_eos_token=True or False are both ok
                                        batch_size=args.eval_batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="tail") if valid_dataset else None

    test_dataloader = PromptDataLoader(dataset=test_dataset, template=model_or_prompt_tuning_template, verbalizer=model_or_prompt_tuning_verbalizer, tokenizer=model_or_prompt_tuning_plm_tokenizer,
                                       tokenizer_wrapper_class=model_or_prompt_tuning_plm_wrapper_class, max_seq_length=args.max_seq_l, decoder_max_length=3,
                                       batch_size=args.eval_batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                       truncate_method="tail") if test_dataset else None

    logger.info("truncate rate: {}".format(
        test_dataloader.tokenizer_wrapper.truncate_rate))
    #================================================================================#
    # load optimizer & scheduler
    if args.mode == 'prompt_tuning':
        optimizer1 = None
        scheduler1 = None
        optimizer_grouped_parameters2 = [{'params': [p for name, p in model_or_prompt_tuning_prompt_model.template.named_parameters(
        ) if 'raw_embedding' not in name]}]  # note that you have to remove the raw_embedding manually from the optimization
        if args.optimizer.lower() == "adafactor":
            optimizer2 = Adafactor(optimizer_grouped_parameters2,
                                   lr=args.prompt_tuning_lr,
                                   relative_step=False,
                                   scale_parameter=False,
                                   warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            scheduler2 = get_constant_schedule_with_warmup(
                optimizer2, num_warmup_steps=args.warmup_step_prompt_tuning)
        elif args.optimizer.lower() == "adamw":
            optimizer2 = AdamW(optimizer_grouped_parameters2,
                               lr=args.prompt_tuning_lr)  # usually lr = 0.5
            scheduler2 = get_linear_schedule_with_warmup(
                optimizer2,
                num_warmup_steps=args.warmup_step_prompt_tuning, num_training_steps=args.tuning_max_steps)  # usually num_warmup_steps is 500
    elif args.mode == 'fine_tuning' or args.mode == 'distillation':
        # it's always good practice to set no decay to biase and LayerNorm parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in model_or_prompt_tuning_prompt_model.plm.named_parameters(
            ) if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.0},
            {'params': [p for n, p in model_or_prompt_tuning_prompt_model.plm.named_parameters(
            ) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters1,
                           lr=args.fine_tuning_lr)
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1,
            num_warmup_steps=args.warmup_step_model_tuning, num_training_steps=args.tuning_max_steps)
        optimizer2 = None
        scheduler2 = None
    #================================================================================#
    # train
    tot_step = args.tuning_max_steps
    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    glb_step = 0
    actual_step = 0
    leave_training = False

    acc_traces = []
    tot_train_time = 0
    pbar_update_freq = 10
    model_or_prompt_tuning_prompt_model.train()

    pbar = tqdm(total=tot_step, desc="Train")
    epoch = 0
    while True:
        pbar.set_description(f"epoch {epoch}")
        # logger.info(f"Begin epoch {epoch}")
        for step, inputs in enumerate(train_dataloader):
            if args.use_cuda:
                inputs = inputs.cuda()
            tot_train_time -= time.time()
            loss = model_or_prompt_tuning_prompt_model(inputs)
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1

            if actual_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model_or_prompt_tuning_prompt_model.parameters(), 1.0)
                glb_step += 1
                if glb_step % pbar_update_freq == 0:
                    aveloss = (tot_loss - log_loss)/pbar_update_freq
                    pbar.update(10)
                    pbar.set_postfix({'loss': aveloss})
                    writer.add_scalar("tuning_loss", aveloss,
                                      global_step=glb_step)
                    log_loss = tot_loss

                if optimizer1 is not None:
                    optimizer1.step()
                    optimizer1.zero_grad()
                if scheduler1 is not None:
                    scheduler1.step()
                if optimizer2 is not None:
                    optimizer2.step()
                    optimizer2.zero_grad()
                if scheduler2 is not None:
                    scheduler2.step()

                tot_train_time += time.time()

                if glb_step % args.tuning_eval_every_steps == 0:
                    val_acc = evaluate(args, model_or_prompt_tuning_prompt_model, valid_dataloader)
                    writer.add_scalar("tuning_val_acc",
                                      val_acc, global_step=glb_step)
                    if val_acc >= best_val_acc:
                        # []TODO 保存路径需要根据根目录修改
                        torch.save(
                            model_or_prompt_tuning_prompt_model.state_dict(), os.path.join(tuning_model_save_path, 'best.ckpt'))
                        best_val_acc = val_acc

                    acc_traces.append(val_acc)
                    logger.info("Glb_step {}, val_acc {}, average time {}".format(
                        glb_step, val_acc, tot_train_time/actual_step))
                    model_or_prompt_tuning_prompt_model.train()

            if glb_step > args.tuning_max_steps:
                leave_training = True
                break

        if leave_training:
            break
        epoch += 1

    thres99 = 0.99*best_val_acc
    thres98 = 0.98*best_val_acc
    thres100 = best_val_acc
    step100 = step98 = step99 = args.tuning_max_steps
    for val_time, acc in enumerate(acc_traces):
        if acc >= thres98:
            step98 = min(val_time*args.tuning_eval_every_steps, step98)
            if acc >= thres99:
                step99 = min(val_time*args.tuning_eval_every_steps, step99)
                if acc >= thres100:
                    step100 = min(val_time*args.tuning_eval_every_steps, step100)

    content_write = f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}"
    logger.info(content_write)

    writer.add_scalar("tuning_best_val_acc", best_val_acc)
    #================================================================================#
    # test
    model_or_prompt_tuning_prompt_model.load_state_dict(
            torch.load(os.path.join(tuning_model_save_path, 'best.ckpt')))
    model_or_prompt_tuning_prompt_model.eval()
    test_acc = evaluate(args, model_or_prompt_tuning_prompt_model, test_dataloader,True)
    logger.info(f"TestAcc: {test_acc}")
    writer.add_scalar("tuning_test_acc", test_acc)
    #================================================================================#


    #=============================distillation=======================================#
    if args.mode == 'distillation':
        #================================================================================#
        # 预处理
        logger.info("begain distillation")
        distill_model_save_path = os.path.join(EXP_PATH, 'distillation_ckpt')
        if not os.path.exists(distill_model_save_path):
            os.mkdir(distill_model_save_path)
        distillation_plm_model, distillation_plm_tokenizer, distillation_plm_config, distillation_plm_wrapper_class = load_plm(
            args.model_name, args.model_path)
        if args.use_cuda:
            distillation_plm_model = distillation_plm_model.cuda()
        if args.model_parallelize:
            distillation_plm_model.parallelize()
        #================================================================================#
        # plm，verbalizer，prompt model
        distillation_template = SoftTemplate(model=distillation_plm_model, tokenizer=distillation_plm_tokenizer, num_tokens=args.soft_token_num,
                                             initialize_from_vocab=args.init_from_vocab).from_file(args.template_path, choice=args.template_id)
        # distillation_verbalizer = ManualVerbalizer(distillation_plm_tokenizer,classes=Processor.get_labels()).from_file(args.manual_verbalizer_path, choice=args.manual_verbalizer_id)
        distillation_verbalizer = GenerationVerbalizer(distillation_plm_tokenizer, classes=Processor.get_labels(
        ), is_rule=True).from_file(args.manual_verbalizer_path, choice=args.manual_verbalizer_id)
        distillation_prompt_model = PromptForGeneration(
            distillation_plm_model, distillation_template, freeze_plm=True)
        #================================================================================#
        # optimizer & scheduler
        optimizer_grouped_parameters = [{'params': [p for name, p in distillation_prompt_model.template.named_parameters(
        ) if 'raw_embedding' not in name]}]  # note that you have to remove the raw_embedding manually from the optimization
        if args.optimizer.lower() == "adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=args.prompt_tuning_lr,
                                  relative_step=False,
                                  scale_parameter=False,
                                  warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_step_prompt_tuning)
        elif args.optimizer.lower() == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.prompt_tuning_lr)  # usually lr = 0.5
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_step_prompt_tuning, num_training_steps=args.distill_max_steps)  # usually num_warmup_steps is 500
        #================================================================================#
        # load FT model & distiller
        model_or_prompt_tuning_prompt_model.load_state_dict(
            torch.load(os.path.join(tuning_model_save_path, 'best.ckpt')))
        model_or_prompt_tuning_prompt_model.eval()

        distiller = Distiller(
            args, model_or_prompt_tuning_prompt_model, distillation_prompt_model)
        #================================================================================#
        # train
        logger.info("Begin distillation.")

        tot_step = args.distill_training_total_steps
        tot_loss = 0
        log_loss = 0
        best_val_acc = 0
        glb_step = 0
        actual_step = 0
        leave_training = False

        acc_traces = []
        tot_train_time = 0
        pbar_update_freq = 10
        distillation_prompt_model.train()

        pbar = tqdm(total=tot_step, desc="Train")
        epoch = 0
        while True:
            pbar.set_description(f"epoch {epoch}")
            # logger.info(f"Begin epoch {epoch}")
            for step, inputs in enumerate(train_dataloader):
                if args.use_cuda:
                    inputs = inputs.cuda()
                tot_train_time -= time.time()
                loss = distiller(inputs,glb_step)
                loss.backward()
                tot_loss += loss.item()
                actual_step += 1

                if actual_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        distillation_prompt_model.parameters(), 1.0)
                    glb_step += 1
                    if glb_step % pbar_update_freq == 0:
                        aveloss = (tot_loss - log_loss)/pbar_update_freq
                        pbar.update(10)
                        pbar.set_postfix({'loss': aveloss})
                        writer.add_scalar(
                            "distill_loss", aveloss, global_step=glb_step)
                        log_loss = tot_loss

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    tot_train_time += time.time()

                    if glb_step % args.distill_eval_every_steps == 0:
                        val_acc = evaluate(args, distillation_prompt_model, valid_dataloader)
                        writer.add_scalar("distill_val_acc",
                                          val_acc, global_step=glb_step)
                        if val_acc >= best_val_acc:
                            # []TODO
                            torch.save(
                                distillation_prompt_model.state_dict(), os.path.join(distill_model_save_path, 'best.ckpt'))
                            best_val_acc = val_acc

                        acc_traces.append(val_acc)
                        logger.info("Glb_step {}, val_acc {}, average time {}".format(
                            glb_step, val_acc, tot_train_time/actual_step))
                        distillation_prompt_model.train()

                if glb_step > args.distill_training_total_steps:
                    leave_training = True
                    break

            if leave_training:
                break
            epoch += 1

        thres99 = 0.99*best_val_acc
        thres98 = 0.98*best_val_acc
        thres100 = best_val_acc
        step100 = step98 = step99 = args.distill_max_steps
        for val_time, acc in enumerate(acc_traces):
            if acc >= thres98:
                step98 = min(val_time*args.distill_eval_every_steps, step98)
                if acc >= thres99:
                    step99 = min(val_time*args.distill_eval_every_steps, step99)
                    if acc >= thres100:
                        step100 = min(val_time*args.distill_eval_every_steps, step100)

        content_write = f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}"
        logger.info(content_write)

        writer.add_scalar("distill_best_val_acc", best_val_acc)
        #================================================================================#
        #test
        distillation_prompt_model.load_state_dict(
            torch.load(os.path.join(distill_model_save_path, 'best.ckpt')))
        distillation_prompt_model.eval()
        test_acc = evaluate(args, distillation_prompt_model, test_dataloader)
        logger.info(f"TestAcc: {test_acc}")
        writer.add_scalar("distill_test_acc", test_acc)

        
    if args.save_ckpt == False:
        os.unlink(os.path.join(tuning_model_save_path, 'best.ckpt'))
        if args.mode == 'distillation':
            os.unlink(os.path.join(distill_model_save_path, 'best.ckpt'))
    return test_acc

if __name__ == '__main__':
    # import sys
    # print(sys.argv)
    # sys.argv = ['exp/main.py', '--use_cuda', 'True', '--model_parallelize', 'False', '--learning_setting', 'few_shot', '--mode', 'fine_tuning', '--dataset_name', 'super_glue.boolq', '--fewshot_seeds', '1', '2', '3', '--few_shot_num_examples_per_label', '16', '--few_shot_also_sample_dev', 'True', '--few_shot_num_examples_per_label_dev', '16', '--model_name', 't5', '--model_path', 't5-large', '--template_path', 'soft_template.txt', '--manual_verbalizer_path', 'manual_verbalizer.txt', '--train_batch_size', '4', '--eval_batch_size', '8', '--max_steps', '20', '--gradient_accumulation_steps', '1', '--eval_every_steps', '20']
    #================================================================================#
    # preprocess
    args = get_my_args()
    
    EXP_PATH = get_experiment_dir(args)
    save_args(args, EXP_PATH)
    init_logger(os.path.join(EXP_PATH, "log.txt"), logging.INFO, logging.INFO)
    logger.info(f'EXP_PATH: {EXP_PATH}')
    set_seed(args.seed)
    #================================================================================#
    # dataset
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(
        args, test=(args.test == True or args.learning_setting == 'zero_shot'))
    if args.replace_test_by_valid:
        logger.info("replaced test by valid")
        test_dataset = valid_dataset
    #================================================================================#
    # main
    if args.learning_setting == 'full':
        logger.info(
            f'dataset len: train:{len(train_dataset)}, valid:{len(valid_dataset)}, test:{len(test_dataset)}')
        res = trainer(
            EXP_PATH,
            args,
            Processor,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )
    elif args.learning_setting == 'few_shot':

        seeds = args.fewshot_seeds
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label=args.few_shot_num_examples_per_label,
                    also_sample_dev=args.few_shot_also_sample_dev,
                    num_examples_per_label_dev=args.few_shot_num_examples_per_label_dev
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset=train_dataset,
                    valid_dataset=train_dataset,
                    seed=seed
                )

                valid_dataset_used = valid_sampled_dataset

                test_dataset = valid_dataset
                # if args.auto_shorten_valid_dataset:
                #     if len(valid_dataset) > 300:
                #         valid_dataset_used = valid_sampled_dataset
                #     else:
                #         valid_dataset_used = valid_dataset
                # else:
                #     valid_dataset_used = valid_sampled_dataset

                logger.info(
                    f'dataset len: train:{len(train_sampled_dataset)}, valid:{len(valid_dataset_used)}, test:{len(test_dataset)}')

                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    args,
                    Processor,
                    train_dataset=train_sampled_dataset,
                    valid_dataset=valid_dataset_used,
                    test_dataset=test_dataset
                )
            else:
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    args,
                    Processor,
                    test_dataset=test_dataset,
                )
            res += result
        res /= len(seeds)
    elif args.learning_setting == 'zero_shot':
        res = trainer(
            EXP_PATH,
            args,
            Processor,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    print("metric:", res)

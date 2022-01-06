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

def get_my_args():
    parser = argparse.ArgumentParser(
        "Global Config Argument Parser", allow_abbrev=False)
    parser.add_argument("--config_yaml_distillation", type=str,
                        help='the configuration file prompt for this experiment.')
    parser.add_argument("--config_yaml_model_tuning", required=True, type=str,
                        help='the configuration file model  for this experiment.')
    parser.add_argument("--resume_model_tuning", type=str, help='a specified logging path to resume training.\
           It will fall back to run from initialization if no lastest checkpoint are found.')
    parser.add_argument("--test", type=str,
                        help='a specified logging path to test')
    parser.add_argument("--resume", type=str)
    parser.add_argument("--replace_test_by_valid", action='store_true')
    parser.add_argument("--auto_shorten_valid_dataset", action='store_false')
    parser.add_argument("--mode",required=True,type=str,choices=['prompt_tuning','model_tuning','distillation'])
    parser.add_argument("--dataset_name_", default='None',type=str)
    parser.add_argument("--dataset_path_", default='None',type=str)
    parser.add_argument("--template_path_", default='None',type=str)
    parser.add_argument("--manual_verbalizer_path_", default='None',type=str)
    args = parser.parse_args()
    return args

def get_my_config(args,mode):

    config2 = get_user_config(args.config_yaml_model_tuning)
    check_config_conflicts(config2)
    if mode=='distillation':
        config1 = get_user_config(args.config_yaml_distillation)
        check_config_conflicts(config1)
        return config1, config2
    return config2, None


def build_dataloader(dataset, template, tokenizer, tokenizer_wrapper_class, config, split):
    dataloader = PromptDataLoader(
        dataset=dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        batch_size=config[split].batch_size,
        shuffle=config[split].shuffle_data,
        teacher_forcing=config[split].teacher_forcing if hasattr(
            config[split], 'teacher_forcing') else None,
        predict_eos_token=True if config.task == "generation" else False,
        **config.dataloader
    )
    return dataloader




def distribution_loss(prompt, model):
    return torch.sum((prompt-model)**2)


def cos_loss(input1, input2, margin=0.1, reduction='mean'):
    batch_size = input1.shape[0]
    target = torch.ones((batch_size,), device=input1.device)
    return F.cosine_embedding_loss(input1.view(batch_size, -1), input2.view(batch_size, -1), target, margin=margin, reduction=reduction)


def soft_loss(pred, label, T=1):
    assert pred.shape == label.shape
    label = label.to(pred.device)
    s = F.softmax(pred/T, dim=1)
    t = F.softmax(label/T, dim=1)
    L = torch.sum(-t*torch.log(s), dim=1)
    return torch.sum(L)/(pred.shape[0])*T*T



class Distillation_classification_Runner(BaseRunner):
    def __init__(self,
                 prompt_tuning_model: PromptForClassification,
                 model_tuning_model: PromptForClassification,
                 config: CfgNode = None,
                 model_tuning_config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 loss_function: Optional[Callable] = None,
                 id2label: Optional[Dict] = None,
                 ):
        super().__init__(model=prompt_tuning_model,
                         config=config,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         )
        self.model_tuning_model = model_tuning_model
        for p in model_tuning_model.parameters():
            p.requires_grad = False
        self.prompt_tuning_model = prompt_tuning_model
        self.loss_function1 = loss_function
        self.loss_function2 = torch.nn.CrossEntropyLoss()
        self.id2label = id2label
        self.label_path_sep = config.dataset.label_path_sep
        self.distillation_step = 0
        self.frozen_batch = None
        self.frozen_batch1 = None

    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()

    def inference_epoch_end(self, split, outputs):
        self.model_tuning_model.eval()
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred)
            labels.extend(label)
        self.save_results(split, {
            'preds': preds,
            'labels': labels,
        })
        metrics = OrderedDict()
        for metric_name in self.config.classification.metric:
            metric = classification_metrics(
                preds, labels, metric_name, id2label=self.id2label, label_path_sep=self.label_path_sep)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        self.global_step
        self.distillation_step += 1

        if self.global_step == self.config.train.acceleration_num_training_steps+1:
            logger.info('Begin prompt tuning only.')

        # model model output
        if self.global_step <= self.config.train.acceleration_num_training_steps:

            outputs2 = self.model_tuning_model.prompt_model(batch)
            # hidden_state_at_mask2 = outputs2['decoder_hidden_states'][-1][:,1,:]
            # hidden_state_at_mask2 = self.model_tuning_model.extract_at_mask(
            #     outputs2['decoder_hidden_states'][-1], batch)
            
            logits2 = self.model_tuning_model.verbalizer.gather_outputs(outputs2)
            outputs_at_mask2 = self.model_tuning_model.extract_at_mask(
                logits2, batch)
            label_words_logits2 = self.model_tuning_model.verbalizer.process_outputs(
                outputs_at_mask2, batch=batch)

        
        # prompt model output
        outputs1 = self.prompt_tuning_model.prompt_model(batch)
        # outputs_at_mask1 = outputs1['decoder_hidden_states'][-1][:,1,:]
        # hidden_state_at_mask1 = self.prompt_tuning_model.extract_at_mask(
        #     outputs1['decoder_hidden_states'][-1], batch)
        logits1 = self.prompt_tuning_model.verbalizer.gather_outputs(outputs1)
        outputs_at_mask1 = self.prompt_tuning_model.extract_at_mask(
            logits1, batch)
        label_words_logits1 = self.prompt_tuning_model.verbalizer.process_outputs(
            outputs_at_mask1, batch=batch)
        
        if self.global_step % 100 == 1:
            if self.global_step <= self.config.train.acceleration_num_training_steps: 
                print('\nmodel',F.softmax(label_words_logits2,dim=1),flush=True)
            print('\nprompt',F.softmax(label_words_logits1,dim=1),flush=True)

        # hard loss
        loss1 = self.loss_function2(label_words_logits1, batch['label'])

        if self.global_step <= self.config.train.acceleration_num_training_steps:  
            # soft loss
            loss2 = soft_loss(label_words_logits1, label_words_logits2.detach(), self.config.train.temperature).to(loss1.device)
            # hidden state loss
            loss3 = self.get_loss_some_hidden(outputs1,outputs2,batch).to(loss1.device)
            
        if self.global_step <= self.config.train.acceleration_num_training_steps:
            return self.config.train.alpha*loss1 + (self.config.train.beta*loss2 + self.config.train.gamma*loss3)# (1-self.distillation_step/self.config.train.acceleration_num_training_steps)
        else:
            return loss1

    def get_loss_last_hidden(self,distillation_outputs,model_tuning_outputs,batch):

        distillation_hidden_state_at_mask = self.prompt_tuning_model.extract_at_mask(
            distillation_outputs['decoder_hidden_states'][-1], batch)
        model_tuning_hidden_state_at_mask = self.model_tuning_model.extract_at_mask(
                model_tuning_outputs['decoder_hidden_states'][-1], batch)
        
        return F.mse_loss(distillation_hidden_state_at_mask,model_tuning_hidden_state_at_mask)/batch.shape[0]

    def get_loss_some_hidden(self,distillation_outputs,model_tuning_outputs,batch,idx_hidden_state_nedd=None):
        
        batch_size = distillation_outputs['decoder_hidden_states'][-1].shape[0]
        if idx_hidden_state_nedd==None:
            idx_hidden_state_need = list(range(12,25,2))
        device_to = distillation_outputs['decoder_hidden_states'][idx_hidden_state_need[0]].device

        distillation_hidden_state_at_mask = [self.prompt_tuning_model.extract_at_mask(
            distillation_outputs['decoder_hidden_states'][i], batch).to(device_to) for i in idx_hidden_state_need]
        distillation_hidden_state_at_mask = torch.stack(distillation_hidden_state_at_mask,dim=1)

        model_tuning_hidden_state_at_mask = [self.model_tuning_model.extract_at_mask(
                model_tuning_outputs['decoder_hidden_states'][i], batch).to(device_to) for i in idx_hidden_state_need]
        model_tuning_hidden_state_at_mask = torch.stack(model_tuning_hidden_state_at_mask,dim=1)

        return F.mse_loss(distillation_hidden_state_at_mask,model_tuning_hidden_state_at_mask)/len(idx_hidden_state_need)/batch_size

        


        

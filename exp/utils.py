import argparse
import os
import time
from openprompt.data_utils import PROCESSORS
from openprompt.utils.logging import logger
from openprompt import PromptDataLoader
import torch
import torch.nn.functional as F
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition maybe?
def get_my_args():
    parser = argparse.ArgumentParser("Global Config Argument Parser", allow_abbrev=False)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument("--model_parallelize", type=t_or_f, default=True)
    parser.add_argument("--test", type=t_or_f, default=False)
    parser.add_argument("--replace_test_by_valid", type=t_or_f, default=False)
    parser.add_argument("--use_cuda", type=t_or_f, default=True)
    parser.add_argument("--learning_setting", type=str, choices=['full','few_shot','zero_shot'],required=True)
    parser.add_argument("--resume_model_tuning", type=str)
    parser.add_argument("--mode",required=True,type=str,choices=['prompt_tuning','fine_tuning','distillation'])
    parser.add_argument("--save_ckpt", type=t_or_f, default=True)

    parser.add_argument("--dataset_name", type=str,required=True)
    parser.add_argument("--dataset_path", type=str)

    parser.add_argument('--fewshot_seeds', nargs='+', type=int,default=[0])
    parser.add_argument('--few_shot_num_examples_per_label', type=int,default=16)
    parser.add_argument('--few_shot_also_sample_dev', type=t_or_f, default=True)
    parser.add_argument('--few_shot_num_examples_per_label_dev', type=int,default=16)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--soft_token_num", type=int,default=20)
    parser.add_argument("--init_from_vocab", type=t_or_f, default=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--template_id", type=int, default=0)
    parser.add_argument("--manual_verbalizer_path", type=str, required=True)
    parser.add_argument("--manual_verbalizer_id", type=int, default=0)


    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_seq_l", type=int, default=480)
    # parser.add_argument("--dataset_decoder_max_length", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="adafactor")
    # parser.add_argument("--warmup_step_prompt", type=int, default=0)

    parser.add_argument("--prompt_tuning_lr", type=float, default=0.3)
    parser.add_argument("--fine_tuning_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_step_prompt_tuning", type=int, default=500)
    parser.add_argument("--warmup_step_model_tuning", type=int, default=500)
    parser.add_argument("--tuning_max_steps", type=int, required=True) # ft 1000, pt 10000
    parser.add_argument("--distill_max_steps", type=int)# dl 5000
    parser.add_argument("--distill_training_total_steps", type=int)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--tuning_eval_every_steps", type=int, default=500)
    parser.add_argument("--distill_eval_every_steps", type=int, default=1000)
    # parser.add_argument("--model_save_path", type=int, required=True)

    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    # parser.add_argument("--warmup_step_model_tuning", type=int, default=32)
    parser.add_argument("--temperature", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)



    # parser.add_argument("--auto_shorten_valid_dataset", action='store_false')
    args = parser.parse_args()

    return args

def get_experiment_dir(args):
    fpath = os.path.split(os.path.realpath(__file__))[0]
    log_path = os.path.join(fpath,'..','log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    now_time = time.strftime("%m_%d-%H_%M_%S", time.localtime())
    typical_name = f"{args.mode}_{args.dataset_name}_{args.learning_setting}_seed{args.seed}_{now_time}"
    exp_path = os.path.join(log_path,typical_name)
    os.mkdir(exp_path)

    # args.EXP_PATH = exp_path
    return exp_path

def save_args(args,EXP_PATH):
    '''
    保存args到configs.txt
    '''
    dic = args.__dict__
    with open(os.path.join(EXP_PATH,"configs.txt"), 'w') as f:
        for key in dic:
            f.write("{}\t{}\n".format(key,dic[key]))

def load_dataset(args, return_class=True, test=False):

    processor = PROCESSORS[(args.dataset_name).lower()]()

    train_dataset = None
    valid_dataset = None
    print(args.dataset_path)
    if not test:
        try:
            train_dataset = processor.get_train_examples(args.dataset_path)
        except FileNotFoundError:
            logger.warning(f"Has no training dataset in {args.dataset_path}.")
        try:
            valid_dataset = processor.get_dev_examples(args.dataset_path)
        except FileNotFoundError:
            logger.warning(f"Has no validation dataset in {args.dataset_path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(args.dataset_path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {args.dataset_path}.")

    # checking whether donwloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset

# def build_dataloader(dataset, template, tokenizer, tokenizer_wrapper_class, config, split):
#     dataloader = PromptDataLoader(
#         dataset=dataset,
#         template=template,
#         tokenizer=tokenizer,
#         tokenizer_wrapper_class=tokenizer_wrapper_class,
#         batch_size=config[split].batch_size,
#         shuffle=config[split].shuffle_data,
#         teacher_forcing=config[split].teacher_forcing if hasattr(
#             config[split], 'teacher_forcing') else None,
#         predict_eos_token=True if config.task == "generation" else False,
#         **config.dataloader
#     )
#     return dataloader
def soft_loss(pred, label, T=1):
    # print(pred.shape)
    assert pred.shape == label.shape
    label = label.to(pred.device)
    s = F.softmax(pred/T, dim=-1)
    t = F.softmax(label/T, dim=-1)
    L = torch.sum(-t*torch.log(s), dim=-1)
    return torch.sum(L)/(pred.shape[0])*T*T

class Distiller(torch.nn.Module):
    def __init__(self,
                args,
                fine_tuning_model,
                distillation_model
                 ):
        # super().__init__(model=prompt_tuning_model,
        #                  config=config,
        #                  train_dataloader=train_dataloader,
        #                  valid_dataloader=valid_dataloader,
        #                  test_dataloader=test_dataloader,
        #                  )
        super(Distiller,self).__init__()
        self.args = args
        self.distillation_model = distillation_model
        self.fine_tuning_model = fine_tuning_model
        self.tag =False
        for p in fine_tuning_model.parameters():
            p.requires_grad = False
        
        print(f"fine_tuning_model.parameters(): {fine_tuning_model.parameters()}")

        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.label_path_sep = config.dataset.label_path_sep
        self.distillation_step = 0
        self.frozen_batch = None
        self.frozen_batch1 = None

    # def inference_step(self, batch, batch_idx):
    #     label = batch.pop('label')
    #     logits = self.model(batch)
    #     pred = torch.argmax(logits, dim=-1)
    #     return pred.cpu().tolist(), label.cpu().tolist()

    # def inference_epoch_end(self, split, outputs):
    #     self.model_tuning_model.eval()
    #     preds = []
    #     labels = []
    #     for pred, label in outputs:
    #         preds.extend(pred)
    #         labels.extend(label)
    #     self.save_results(split, {
    #         'preds': preds,
    #         'labels': labels,
    #     })
    #     metrics = OrderedDict()
    #     for metric_name in self.config.classification.metric:
    #         metric = classification_metrics(
    #             preds, labels, metric_name, id2label=self.id2label, label_path_sep=self.label_path_sep)
    #         metrics[metric_name] = metric
    #     return metrics
    def shift_logits_and_labels(self, 
                                logits, 
                                loss_ids, 
                                reference_ids):
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous().bool()
        shift_input_ids = reference_ids[..., 1:].contiguous()
        # shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100) 
        return shift_logits[shift_loss_ids], shift_input_ids[shift_loss_ids]
    def shift_logits(self, 
                                logits, 
                                loss_ids, 
                                ):
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous().bool()
        # shift_input_ids = reference_ids[..., 1:].contiguous()
        # shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100) 
        return shift_logits[shift_loss_ids]
    def forward(self, batch, global_step):
        # print(batch['loss_ids'])
        global_step
        self.distillation_step += 1

        if self.fine_tuning_model.config.is_encoder_decoder:
            reference_ids = batch['decoder_input_ids']
        else:
            reference_ids = batch['input_ids']

        if global_step == self.args.distill_max_steps+1:
            logger.info('Begin prompt tuning only.')

        # model model output
        if global_step <= self.args.distill_max_steps:

            outputs2 = self.fine_tuning_model.prompt_model(batch)
            
            # hidden_state_at_mask2 = outputs2['decoder_hidden_states'][-1][:,1,:]
            # hidden_state_at_mask2 = self.model_tuning_model.extract_at_mask(
            #     outputs2['decoder_hidden_states'][-1], batch)
            
            logits2 = outputs2.logits
            label_words_logits2, labels2 = self.shift_logits_and_labels(logits2, batch['loss_ids'], reference_ids)
            # outputs_at_mask2 = self.fine_tuning_model.extract_at_mask(
            #     logits2, batch)
            # label_words_logits2 = self.fine_tuning_model.verbalizer.process_outputs(
            #     outputs_at_mask2, batch=batch)

        
        # prompt model output
        outputs1 = self.distillation_model.prompt_model(batch)
        # outputs_at_mask1 = outputs1['decoder_hidden_states'][-1][:,1,:]
        # hidden_state_at_mask1 = self.prompt_tuning_model.extract_at_mask(
        #     outputs1['decoder_hidden_states'][-1], batch)
        logits1 = outputs1.logits
        label_words_logits1, labels1 = self.shift_logits_and_labels(logits1, batch['loss_ids'], reference_ids)
        # outputs_at_mask1 = self.distillation_model.extract_at_mask(
        #     logits1, batch)
        # label_words_logits1 = self.distillation_model.verbalizer.process_outputs(
        #     outputs_at_mask1, batch=batch)
        

        # hard loss
        loss1 = self.loss_function(label_words_logits1, labels1)

        if global_step <= self.args.distill_max_steps:  
            # soft loss
            loss2 = soft_loss(label_words_logits1, label_words_logits2.detach(), self.args.temperature).to(loss1.device)
            # hidden state loss
            loss3 = self.get_loss_some_hidden(outputs1,outputs2,batch).to(loss1.device)
            
        if global_step <= self.args.distill_max_steps:
            return self.args.alpha*loss1 + (self.args.beta*loss2 + self.args.gamma*loss3)# (1-self.distillation_step/self.config.train.acceleration_num_training_steps)
        else:
            return loss1

    def get_loss_last_hidden(self,distillation_outputs,model_tuning_outputs,batch):

        distillation_hidden_state_at_mask = self.shift_logits(
            distillation_outputs['decoder_hidden_states'][-1], batch['loss_ids'])
        model_tuning_hidden_state_at_mask = self.shift_logits(
                model_tuning_outputs['decoder_hidden_states'][-1], batch['loss_ids'])
        
        return F.mse_loss(distillation_hidden_state_at_mask,model_tuning_hidden_state_at_mask)/batch.shape[0]

    def get_loss_some_hidden(self,distillation_outputs,model_tuning_outputs,batch,idx_hidden_state_need=None):
        if self.tag == False:
            print(distillation_outputs['decoder_hidden_states'][0].shape,len(distillation_outputs['decoder_hidden_states']))
            self.tag=True
        batch_size = distillation_outputs['decoder_hidden_states'][-1].shape[0]
        if idx_hidden_state_need==None:
            idx_hidden_state_need = list(range(12,25,2))
        device_to = distillation_outputs['decoder_hidden_states'][idx_hidden_state_need[0]].device

        distillation_hidden_state_at_mask = [self.shift_logits(
            distillation_outputs['decoder_hidden_states'][i], batch['loss_ids']).to(device_to) for i in idx_hidden_state_need]
        distillation_hidden_state_at_mask = torch.stack(distillation_hidden_state_at_mask,dim=1)

        model_tuning_hidden_state_at_mask = [self.shift_logits(
                model_tuning_outputs['decoder_hidden_states'][i], batch['loss_ids']).to(device_to) for i in idx_hidden_state_need]
        model_tuning_hidden_state_at_mask = torch.stack(model_tuning_hidden_state_at_mask,dim=1)

        return F.mse_loss(distillation_hidden_state_at_mask,model_tuning_hidden_state_at_mask)/len(idx_hidden_state_need)/batch_size
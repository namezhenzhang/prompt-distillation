from openprompt.data_utils import PROCESSORS
processor = PROCESSORS[('super_glue.boolq').lower()]()
print(PROCESSORS)
# train_dataset = processor.get_train_examples(None)
# test_dataset = processor.get_test_examples(None)
# print('h')

{'fewnerd': <class 'openprompt.data_utils.typing_dataset.FewNERDProcessor'>, 'agnews': <class 'openprompt.data_utils.text_classification_dataset.AgnewsProcessor'>, 'dbpedia': <class 'openprompt.data_utils.text_classification_dataset.DBpediaProcessor'>, 'amazon': <class 'openprompt.data_utils.text_classification_dataset.AmazonProcessor'>, 'imdb': <class 'openprompt.data_utils.text_classification_dataset.ImdbProcessor'>, 'sst-2': <class 'openprompt.data_utils.text_classification_dataset.SST2Processor'>, 'mnli': <class 'openprompt.data_utils.text_classification_dataset.MnliProcessor'>, 'wic': <class 'openprompt.data_utils.fewglue_dataset.WicProcessor'>, 'rte': <class 'openprompt.data_utils.fewglue_dataset.RteProcessor'>, 'cb': <class 'openprompt.data_utils.fewglue_dataset.CbProcessor'>, 'wsc': <class 'openprompt.data_utils.fewglue_dataset.WscProcessor'>, 'boolq': <class 'openprompt.data_utils.fewglue_dataset.BoolQProcessor'>, 'copa': <class 'openprompt.data_utils.fewglue_dataset.CopaProcessor'>, 'multirc': <class 'openprompt.data_utils.fewglue_dataset.MultiRcProcessor'>, 'record': <class 'openprompt.data_utils.fewglue_dataset.RecordProcessor'>, 'tacred': <class 'openprompt.data_utils.relation_classification_dataset.TACREDProcessor'>, 'tacrev': <class 'openprompt.data_utils.relation_classification_dataset.TACREVProcessor'>, 'retacred': <class 'openprompt.data_utils.relation_classification_dataset.ReTACREDProcessor'>, 'semeval': <class 'openprompt.data_utils.relation_classification_dataset.SemEvalProcessor'>, 'LAMA': <class 'openprompt.data_utils.lama_dataset.LAMAProcessor'>, 'webnlg_2017': <class 'openprompt.data_utils.conditional_generation_dataset.WebNLGProcessor'>, 'webnlg': <class 'openprompt.data_utils.conditional_generation_dataset.WebNLGProcessor'>, 'super_glue.multirc': <class 'openprompt.data_utils.huggingface_dataset.SuperglueMultiRCProcessor'>, 'super_glue.boolq': <class 'openprompt.data_utils.huggingface_dataset.SuperglueBoolQProcessor'>, 'super_glue.cb': <class 'openprompt.data_utils.huggingface_dataset.SuperglueCBProcessor'>, 'super_glue.copa': <class 'openprompt.data_utils.huggingface_dataset.SuperglueCOPAProcessor'>, 'super_glue.rte': <class 'openprompt.data_utils.huggingface_dataset.SuperglueRTEProcessor'>, 'super_glue.wic': <class 'openprompt.data_utils.huggingface_dataset.SuperglueWiCProcessor'>, 'super_glue.wsc': <class 'openprompt.data_utils.huggingface_dataset.SuperglueWSCProcessor'>, 'super_glue.record': <class 'openprompt.data_utils.huggingface_dataset.SuperglueRecordProcessor'>, 'snli': <class 'openprompt.data_utils.nli_dataset.SNLIProcessor'>}
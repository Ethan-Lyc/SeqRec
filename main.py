import sys
import os
import logging
import papermill as pm
import scrapbook as sb
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from amazon_reviews import download_and_extract, data_preprocessing
from recommenders.datasets.download_utils import maybe_download


from sli_rec import SLI_RECModel as SeqModel
####  to use the other model, use one of the following lines:
# from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel
# from recommenders.models.deeprec.models.sequential.caser import CaserModel as SeqModel
# from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel as SeqModel
# from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel

#from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel

from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
#from recommenders.models.deeprec.io.nextitnet_iterator import NextItNetIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))
##  ATTENTION: change to the corresponding config file, e.g., caser.yaml for CaserModel, sum.yaml for SUMModel
yaml_file = 'sli_rec.yaml'
EPOCHS = 10
BATCH_SIZE = 400
RANDOM_SEED = SEED  # Set None for non-deterministic result

data_path = os.path.join( "tests", "slirec")
# for test
train_file = os.path.join(data_path, r'train_data')
valid_file = os.path.join(data_path, r'valid_data')
test_file = os.path.join(data_path, r'test_data')
user_vocab = os.path.join(data_path, r'user_vocab.pkl')
item_vocab = os.path.join(data_path, r'item_vocab.pkl')
cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
output_file = os.path.join(data_path, r'output.txt')

reviews_name = 'reviews_Movies_and_TV_5.json'
meta_name = 'meta_Movies_and_TV.json'
reviews_file = os.path.join(data_path, reviews_name)
meta_file = os.path.join(data_path, meta_name)
train_num_ngs = 4 # number of negative instances with a positive instance for training
valid_num_ngs = 4 # number of negative instances with a positive instance for validation
test_num_ngs = 9 # number of negative instances with a positive instance for testing
sample_rate = 0.01 # sample a small item set for training and testing here for fast example

input_files = [reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab]

if not os.path.exists(train_file):
    download_and_extract(reviews_name, reviews_file)
    download_and_extract(meta_name, meta_file)
    data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs)
    #### uncomment this for the NextItNet model, because it does not need to unfold the user history
    # data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs, is_history_expanding=False)
### NOTE:  
### remember to use `_create_vocab(train_file, user_vocab, item_vocab, cate_vocab)` to generate the user_vocab, item_vocab and cate_vocab files, if you are using your own dataset rather than using our demo Amazon dataset.
hparams = prepare_hparams(yaml_file, 
                          embed_l2=0., 
                          layer_l2=0., 
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          show_step=20,
                          MODEL_DIR=os.path.join(data_path, "model/"),
                          SUMMARIES_DIR=os.path.join(data_path, "summary/"),
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=True,
                          train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
            )
input_creator = SequentialIterator
#### uncomment this for the NextItNet model, because it needs a special data iterator for training
#input_creator = NextItNetIterator
model = SeqModel(hparams, input_creator, seed=RANDOM_SEED)

## sometimes we don't want to train a model from scratch
## then we can load a pre-trained model like this: 
#model.load_model(r'tests\slirec\model/best_model')
# test_num_ngs is the number of negative lines after each positive line in your test_file
print(model.run_eval(test_file, num_ngs=test_num_ngs)) 
with Timer() as train_time:
    model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs) 

# valid_num_ngs     is the number of negative lines after each positive line in your valid_file
# we will evaluate the performance of model on valid_file every epoch
print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))
res_syn = model.run_eval(test_file, num_ngs=test_num_ngs)
print(res_syn)

sb.glue("res_syn", res_syn)
model = model.predict(test_file, output_file)
# The data was downloaded in tmpdir folder. You can delete them manually if you do not need them any more.

model_best_trained = SeqModel(hparams, input_creator, seed=RANDOM_SEED)
path_best_trained = os.path.join(hparams.MODEL_DIR, "best_model")
print('loading saved model in {0}'.format(path_best_trained))
model_best_trained.load_model(path_best_trained)
model_best_trained.run_eval(test_file, num_ngs=test_num_ngs)
model_best_trained.predict(test_file, output_file)
from collections import namedtuple

_share_args = ('size',  # input size
               'dataset_type',  # type of dataset
               'dataset_indexes',  # index files for dataset
               'preview_interval',  # interval to save network output for previewing
               'batch_size',  # process batch size
               'seed',  # seed for random number generators
               )

_train_args = ('lr',  # init learning rate
               'pretrained',  # pretrained checkpoint
               'start_epoch',  # start epoch index
               'end_epoch',  # end epoch index (exclusive)
               'sparsity',  # train network with sparsity
               'autocast',  # train with auto mixed precision
               'loss_type',  # loss type for optimization
               'save_path',  # checkpoint save path
               'save_prefix',  # prefix of checkpoint file name
               )

_test_args = ('checkpoints',  # checkpoint to test
              'fp16',  # use fp16 to run network forward
              )

train_arg = namedtuple('train_arg', (*_share_args, *_train_args))
test_arg = namedtuple('test_arg', (*_share_args, *_test_args))

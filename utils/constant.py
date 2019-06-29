import argparse
import torch

parser = argparse.ArgumentParser(description='ASR training')

parser.add_argument('--model', default='TRFS', type=str, help="TRFS:transformer")
parser.add_argument('--name', default='model', help="Name of the model for saving")

# train
parser.add_argument('--train-manifest-list', nargs='+', type=str)
parser.add_argument('--valid-manifest-list', nargs='+', type=str)
parser.add_argument('--test-manifest-list', nargs='+', type=str)
parser.add_argument('--lang-list', nargs='+', type=str)

parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')

parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--label-smoothing', default=0.0, type=float, help='Label smoothing')

parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')

parser.add_argument('--device-ids', default=None, nargs='+', type=int,
                    help='If using cuda, sets the GPU devices for the process')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')

parser.add_argument('--save-every', default=5, type=int, help='Save model every certain number of epochs')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')

parser.add_argument('--emb_trg_sharing', action='store_true', help='Share embedding weight source and target')
parser.add_argument('--feat_extractor', default='vgg_cnn', type=str, help='emb_cnn or vgg_cnn')

parser.add_argument('--verbose', action='store_true', help='Verbose')

parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)

# Transformer
parser.add_argument('--num-layers', default=3, type=int, help='Number of layers')
parser.add_argument('--num-heads', default=5, type=int, help='Number of heads')
parser.add_argument('--dim-model', default=512, type=int, help='Model dimension')
parser.add_argument('--dim-key', default=64, type=int, help='Key dimension')
parser.add_argument('--dim-value', default=64, type=int, help='Value dimension')
parser.add_argument('--dim-input', default=161, type=int, help='Input dimension')
parser.add_argument('--dim-inner', default=1024, type=int, help='Inner dimension')
parser.add_argument('--dim-emb', default=512, type=int, help='Embedding dimension')

parser.add_argument('--src-max-len', default=4000, type=int, help='Source max length')
parser.add_argument('--tgt-max-len', default=1000, type=int, help='Target max length')

# Noam optimizer
parser.add_argument('--warmup', default=4000, type=int, help='Warmup')
parser.add_argument('--min-lr', default=1e-5, type=float, help='min lr')
parser.add_argument('--k-lr', default=1, type=float, help='factor lr')

# SGD optimizer
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--lr-anneal', default=1.1, type=float, help='lr anneal')

# Decoder search
parser.add_argument('--beam-search', action='store_true', help='Beam search')
parser.add_argument('--beam-width', default=3, type=int, help='Beam size')
parser.add_argument('--beam-nbest', default=5, type=int, help='Number of best sequences')
parser.add_argument('--lm-rescoring', action='store_true', help='Rescore using LM')
parser.add_argument('--lm-path', type=str, default="lm_model.pt", help="Path to LM model")
parser.add_argument('--lm-weight', default=0.1, type=float, help='LM weight')
parser.add_argument('--c-weight', default=0.1, type=float, help='Word count weight')
parser.add_argument('--prob-weight', default=1.0, type=float, help='Probability E2E weight')

# loss
parser.add_argument('--loss', type=str, default='ce', help='ce or ctc')
parser.add_argument('--clip', action='store_true', help="clip")
parser.add_argument('--max-norm', default=400, type=float, help="max norm for clipping")

parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')

# Parallelize model
parser.add_argument('--parallel', action='store_true', help='Parallelize the model')

# shuffle
parser.add_argument('--shuffle', action='store_true', help='Shuffle')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

args = parser.parse_args()
USE_CUDA = args.cuda

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

PAD_CHAR = "¶"
SOS_CHAR = "§"
EOS_CHAR = "¤"
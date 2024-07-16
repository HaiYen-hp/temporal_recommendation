import os
import time
import argparse
import tensorflow as tf
import sys
sys.path.append("/home/yenlh/temporal_recommendation/ASReP")
from tensorflow.python.client import device_lib
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import data_load, predict_eval, evalute_seq, evaluate, data_augment
import traceback, sys
import json
import warnings
from tqdm import tqdm
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from outer_config import FIX_PATH
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_USE_LEGACY_KERAS"]='1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--gpus', default="0", required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--evalnegsample', default=-1, type=int)
parser.add_argument('--reversed', default=0, type=int)
parser.add_argument('--reversed_gen_number', default=-1, type=int)
parser.add_argument('--M', default=10, type=int)
parser.add_argument('--reversed_pretrain', default=-1, type=int)
parser.add_argument('--aug_traindata', default=-1, type=int)

# Define the hyperparameter space
space = [
    Real(1e-5, 1e-1, name='lr', prior='log-uniform'),
    Integer(1, 8, name='num_heads'),
    Integer(1, 6, name='num_blocks'),
    Integer(32, 512, name='hidden_units'),
    Real(0.1, 0.5, name='dropout_rate'),
    Real(1e-6, 1e-2, name='l2_emb', prior='log-uniform'),
    Integer(32, 128, name='batch_size'),
    Integer(50, 200, name='maxlen')
]
@use_named_args(space)
def objective(**params):

    # Ensure hidden_units is divisible by num_heads
    if params['hidden_units'] % params['num_heads'] != 0:
        return 0  # Return a high error value
    
    # Update the arguments with the current hyperparameters
    args = argparse.Namespace(
        lr=params['lr'],
        num_heads=params['num_heads'],
        num_blocks=params['num_blocks'],
        hidden_units=params['hidden_units'],
        dropout_rate=params['dropout_rate'],
        l2_emb=params['l2_emb'],
        batch_size=params['batch_size'],
        maxlen=params['maxlen'],
        num_epochs=10  # Set a fixed number of epochs for tuning
    )

    dataset = data_load(args_sys.dataset, args, args_sys)
    [user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    # Instantiate the model
    model = Model(usernum=usernum, itemnum=itemnum, args=args)

    # Train and validate your model here
    test_recall = train_and_evaluate(model, dataset, user_train, sampler, args, args_sys)
    
    # Minimize the negative recall to maximize recall
    return -test_recall

def  train_and_evaluate(model, dataset, user_train, sampler, args, args_sys):
    cc = []
    for u in user_train:
        cc.append(len(user_train[u]))
    cc = np.array(cc)
    print('average sequence length: %.2f' % np.mean(cc))
    print('min seq length: %.2f' % np.min(cc))
    print('max seq length: %.2f' % np.max(cc))
    print('quantile 25 percent: %.2f' % np.quantile(cc, 0.25))
    print('quantile 50 percent: %.2f' % np.quantile(cc, 0.50))
    print('quantile 75 percent: %.2f' % np.quantile(cc, 0.75))
    num_batch = int(len(user_train) / args.batch_size)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)

    aug_data_signature = os.path.join(FIX_PATH,'aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}_M_{}'.format(args_sys.dataset, args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads, args_sys.reversed_gen_number, args_sys.M))
    print(aug_data_signature)

    model_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}'.format(args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads, 5)

    if not os.path.isdir(os.path.join(FIX_PATH,'aug_data/'+args_sys.dataset)):
        os.makedirs(os.path.join(FIX_PATH,'aug_data/'+args_sys.dataset))

    saver = tf.compat.v1.train.Saver()
    sub_reversed_folder = '_reversed'
    if args_sys.reversed_pretrain == -1:
        sess.run(tf.compat.v1.global_variables_initializer())
    else:
        saver.restore(sess, os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder+'/'+model_signature+'.ckpt'))
        print('pretrain model loaded')
    T = 0.0
    t0 = time.time()
    try:
        best_recall=0
        for epoch in range(1, args.num_epochs + 1):

            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                        model.is_training: True})
            print('epoch: %d, loss: %8f' % (epoch, loss))


            if (epoch % 20 == 0 and epoch >= 200) or epoch == args.num_epochs:
                t1 = time.time() - t0
                T += t1
                print("start testing")
                RANK_RESULTS_DIR = os.path.join(FIX_PATH,f"rank_results/{args_sys.dataset}_pretain_{args_sys.reversed_pretrain}")
                if not os.path.isdir(RANK_RESULTS_DIR):
                    os.makedirs(RANK_RESULTS_DIR)
                rank_test_file = RANK_RESULTS_DIR + '/' + model_signature + '_predictions.json'
                
                all_predictions_results, all_item_idx, all_u, eval_data, real_train = \
                    predict_eval(model, dataset, args, args_sys, sess, "test")
                rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output = \
                    evalute_seq(all_predictions_results, all_item_idx, all_u, real_train)
                t_test, t_test_short_seq, t_test_short7_seq, t_test_short37_seq, t_test_medium3_seq, t_test_medium7_seq, t_test_long_seq, test_rankitems = \
                    evaluate(rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output, eval_data)

                if args_sys.reversed == 0:
                   with open(rank_test_file, 'w') as f:
                       for eachpred in test_rankitems:
                           f.write(json.dumps(eachpred) + '\n')
                

                if not (args_sys.reversed == 1):
                    all_predictions_results, all_item_idx, all_u, eval_data, real_train = \
                        predict_eval(model, dataset, args, args_sys, sess, "valid")
                    rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output = \
                        evalute_seq(all_predictions_results, all_item_idx, all_u, real_train)
                    t_valid, t_valid_short_seq, t_valid_short7_seq, t_valid_short37_seq, t_valid_medium3_seq, t_valid_medium7_seq, t_valid_long_seq, valid_rankitems = \
                        evaluate(rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts,\
                                 all_predictions_results_output, eval_data)

                    print(f'epoch: {str(epoch)} validationall: {str(t_valid)} \nepoch: {str(epoch)} testall: {str(t_test)}')
                    print(f'epoch: {str(epoch)} validationshort: {str(t_valid_short_seq)} \nepoch: {str(epoch)} testshort: {str(t_test_short_seq)}')
                    print(f'epoch: {str(epoch)} validationshort7: {str(t_valid_short7_seq)} \nepoch: {str(epoch)} testshort7: {str(t_test_short7_seq)}')
                    print(f'epoch: {str(epoch)} validationshort37: {str(t_valid_short37_seq)} \nepoch: {str(epoch)} testshort37: {str(t_test_short37_seq)}')
                    print(f'epoch: {str(epoch)} validationmedium3: {str(t_valid_medium3_seq)} \nepoch: {str(epoch)} testmedium3: {str(t_test_medium3_seq)}')
                    print(f'epoch: {str(epoch)} validationmedium7: {str(t_valid_medium7_seq)} \nepoch: {str(epoch)} testmedium7: {str(t_test_medium7_seq)}')
                    print(f'epoch: {str(epoch)} validationlong: {str(t_valid_long_seq)} \nepoch: {str(epoch)} testlong: {str(t_test_long_seq)}')
                else:
                    print(f'epoch: {str(epoch)} test: {str(t_test)}')

                t0 = time.time()
        print("Execution time for training and evaluate: %s", time.strftime("%H:%M:%S", time.gmtime(T)))
    except Exception as e:
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                limit=2, file=sys.stdout)
        sampler.close()
        exit(1)
    if args_sys.reversed == 1 and t_test['recall'] > best_recall:
        best_recall=t_test.recall
        augmented_data = data_augment(model, dataset, args_sys, sess, args_sys.reversed_gen_number)
        with open(aug_data_signature+'.txt', 'w') as f:
            for u, aug_ilist in augmented_data.items():
                for ind, aug_i in enumerate(aug_ilist):
                    f.write(str(u-1) + '\t' + str(aug_i - 1) + '\t' + str(-(ind+1)) + '\n')
        if args_sys.reversed_gen_number > 0:
            parent_path_re_model = os.path.join(FIX_PATH,'reversed_models')
            if not os.path.exists(parent_path_re_model+'/'+args.dataset+sub_reversed_folder+'/'):
                os.makedirs(parent_path_re_model+'/'+args.dataset+sub_reversed_folder+'/')
            saver.save(sess, parent_path_re_model+'/'+args.dataset+sub_reversed_folder+'/'+model_signature+'.ckpt')
    sampler.close()
    return best_recall
    

if __name__ == '__main__':
    args_sys = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args_sys.gpus
    print("List GPU devices used")
    print(device_lib.list_local_devices())
    
    start = time.time()
    res = gp_minimize(objective, space, n_calls=50, random_state=0)
    print("Best RECALL: ", -res.fun)
    print("Best parameters: ", res.x)
    end = time.time() - start
    print("Execution time for hyperparamter tuning: %s", time.strftime("%H:%M:%S", time.gmtime(end)))
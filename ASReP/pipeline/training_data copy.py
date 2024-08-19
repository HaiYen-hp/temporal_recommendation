import os
import sys
sys.path.append("/home/yenlh/temporal_recommendation/ASReP")
import time
import argparse
import tensorflow as tf
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
from outer_config import FIX_PATH
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_USE_LEGACY_KERAS"]='1'


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', default="0",required=True)
parser.add_argument('--evalnegsample', default=-1, type=int)
parser.add_argument('--reversed', default=0, type=int)
parser.add_argument('--reversed_gen_number', default=-1, type=int)
parser.add_argument('--M', default=10, type=int)
parser.add_argument('--reversed_pretrain', default=-1, type=int)
parser.add_argument('--aug_traindata', default=-1, type=int)

if __name__ == '__main__':
    args_sys = parser.parse_args()
    with open(os.path.join(FIX_PATH,'best_params.txt'), 'r') as f:
        args = json.load(f)
    print("List GPU devices used")
    print(device_lib.list_local_devices())
    start = time.time()
    dataset = data_load(args_sys.dataset, args, args_sys)
    end = time.time() - start
    print("Execution time for training and evaluate: %s", time.strftime("%H:%M:%S", time.gmtime(end)))

    [user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset
    num_batch = int(len(user_train) / args.batch_size)
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

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = Model(usernum, itemnum, args)
    #sess.run(tf.global_variables_initializer())
    aug_data_signature = './aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}_M_{}'.format(args_sys.dataset, args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads, args_sys.reversed_gen_number, args_sys.M)
    print(aug_data_signature)

    model_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}'.format(args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads, 5)

    if not os.path.isdir(os.path.join(FIX_PATH, 'aug_data/'+args_sys.dataset)):
        os.makedirs(os.path.join(FIX_PATH, 'aug_data/'+args_sys.dataset))

    saver = tf.compat.v1.train.Saver()
    sub_reversed_folder = '_reversed'
    if args.reversed_pretrain == -1:
        sess.run(tf.compat.v1.global_variables_initializer())
    else:
        saver.restore(sess, os.path.join(FIX_PATH, 'reversed_models/'+args_sys.dataset+sub_reversed_folder+'/'+model_signature+'.ckpt'))
        print('pretrain model loaded')

    T = 0.0
    t0 = time.time()

    try:
        for epoch in range(1, args.num_epochs + 1):

            #print(num_batch)
            for step in range(num_batch):#tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            #for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                        model.is_training: True})
            print('epoch: %d, loss: %8f' % (epoch, loss))


            if (epoch % 20 == 0 and epoch >= 200) or epoch == args.num_epochs:
                t1 = time.time() - t0
                T += t1
                print("start testing")
                RANK_RESULTS_DIR = f"./rank_results/{args.dataset}_pretain_{args.reversed_pretrain}"
                if not os.path.isdir(RANK_RESULTS_DIR):
                    os.makedirs(RANK_RESULTS_DIR)
                rank_test_file = RANK_RESULTS_DIR + '/' + model_signature + '_predictions.json'
                
                all_predictions_results, all_item_idx, all_u, eval_data, real_train = \
                    predict_eval(model, dataset, args, args_sys, sess, "test")
                rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output = \
                    evalute_seq(all_predictions_results, all_item_idx, all_u, real_train)
                t_test, t_test_short_seq, t_test_short7_seq, t_test_short37_seq, t_test_medium3_seq, t_test_medium7_seq, t_test_long_seq, test_rankitems = \
                    evaluate(rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output, eval_data)

                del all_predictions_results, all_item_idx, all_u, eval_data, real_train, rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output

                if not (args.reversed == 1):
                    all_predictions_results, all_item_idx, all_u, eval_data, real_train = \
                        predict_eval(model, dataset, args, args_sys, sess, "valid")
                    rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output = \
                        evalute_seq(all_predictions_results, all_item_idx, all_u, real_train)
                    t_valid, t_valid_short_seq, t_valid_short7_seq, t_valid_short37_seq, t_valid_medium3_seq, t_valid_medium7_seq, t_valid_long_seq, valid_rankitems = \
                        evaluate(rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts,\
                                 all_predictions_results_output, eval_data)
                    
                    del all_predictions_results, all_item_idx, all_u, eval_data, real_train, rankeditems_list, test_indices, scale_pred_list, test_allitems, seq_dicts, all_predictions_results_output

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

            print(T)
    except Exception as e:
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                limit=2, file=sys.stdout)
        sampler.close()
        exit(1)

    if args.reversed == 1:
        augmented_data = data_augment(model, dataset, args, sess, args.reversed_gen_number)
        with open(aug_data_signature+'.txt', 'w') as f:
            for u, aug_ilist in augmented_data.items():
                for ind, aug_i in enumerate(aug_ilist):
                    f.write(str(u-1) + '\t' + str(aug_i - 1) + '\t' + str(-(ind+1)) + '\n')
        if args_sys.reversed_gen_number > 0:
            if not os.path.exists(os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder)):
                os.makedirs(os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder))
            saver.save(sess, os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder+'/'+model_signature+'.ckpt'))
    sampler.close()

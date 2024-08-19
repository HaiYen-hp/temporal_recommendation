import os
import time
import argparse
import tensorflow as tf
import sys
sys.path.append("/home/yenlh/temporal_recommendation/ASReP")
from tensorflow.python.client import device_lib
from sampler import WarpSampler
from model import Model
from util import data_load, predict_eval, evalute_seq, evaluate, data_augment, create_spark_session, check_available_memory
import traceback, sys
import json
import warnings
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
from outer_config import FIX_PATH
import socket
import gc
# import pycuda.driver as cuda
# import pycuda.autoinit
import tracemalloc
warnings.filterwarnings("ignore")

tracemalloc.start()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_USE_LEGACY_KERAS"]='1'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['SPARK_DAEMON_JAVA_OPTS'] = "-Xms16g -Xmx64g"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Beauty", required=True)
parser.add_argument('--train_dir', default="0", required=True)
parser.add_argument('--evalnegsample', default=-1, type=int)
parser.add_argument('--reversed', default=0, type=int)
parser.add_argument('--reversed_gen_number', default=-1, type=int)
parser.add_argument('--M', default=10, type=int)
parser.add_argument('--reversed_pretrain', default=-1, type=int)
parser.add_argument('--aug_traindata', default=-1, type=int)

best_recall=np.zeros(3)

def objective(trial):
    global best_recall
    start = time.time()
    dataset = data_load(data_name=args_sys.dataset, args_sys=args_sys, args=None)
    end = time.time() - start
    print("Execution time for load data: %s", time.strftime("%H:%M:%S", time.gmtime(end)))
    
    # TensorFlow session configuration
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()
    tf.compat.v1.disable_eager_execution()
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Available GPUs:", tf.config.experimental.list_physical_devices('GPU'))                  
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.87
    config.allow_soft_placement = True
    
    try:

        hidden_units = trial.suggest_int('hidden_units', 128, 256, step=32)
        num_heads = trial.suggest_int('num_heads', 1, 6)
        # Ensure hidden_units is divisible by num_heads
        hidden_units = hidden_units - (hidden_units % num_heads)
        
        args = {
            'hidden_units': hidden_units,
            'batch_size': 64,
            'num_blocks': trial.suggest_int('num_blocks', 1, 6),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'l2_emb': trial.suggest_float('l2_emb', 1e-6, 1e-2, log=True),
            'num_heads': num_heads,
            'maxlen': 100,
            'num_epochs': 10
        }
        # start = time.time()
        # dataset = data_load(args_sys.dataset, args, args_sys)
        # end = time.time() - start
        # print("Execution time for load data augment: %s", time.strftime("%H:%M:%S", time.gmtime(end)))
            
        [user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset

        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args['batch_size'], maxlen=args['maxlen'], n_workers=3)
        
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
        num_batch = int(len(user_train) / args['batch_size'])
        
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        model = Model(usernum=usernum, itemnum=itemnum, args=args)
        
        aug_data_signature = os.path.join(FIX_PATH,'aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}_M_{}'.format(args_sys.dataset, args['lr'], args['maxlen'], args['hidden_units'], args['num_blocks'], args['dropout_rate'], args['l2_emb'], args['num_heads'], args_sys.reversed_gen_number, args_sys.M))
        print(aug_data_signature)

        model_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_{}'.format(args['lr'], args['maxlen'], args['hidden_units'], args['num_blocks'], args['dropout_rate'], args['l2_emb'], args['num_heads'], 5)
        
        if not os.path.isdir(os.path.join(FIX_PATH,'aug_data/'+args_sys.dataset)):
            os.makedirs(os.path.join(FIX_PATH,'aug_data/'+args_sys.dataset))

        sub_reversed_folder = '_reversed'
        if args_sys.reversed_pretrain == -1:
            sess.run(tf.compat.v1.global_variables_initializer())
        else:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder+'/'+model_signature+'.ckpt'))
            print('pretrain model loaded')

        print("Start training")
        for epoch in range(1, args['num_epochs'] + 1):

            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                            {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                            model.is_training: True})
            print('epoch: %d, loss: %8f' % (epoch, loss))

            if (epoch % 20 == 0 and epoch >= 200) or epoch == args['num_epochs']:
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

        end = time.time() - start
        print("Execution time for training and evaluate %s", time.strftime("%H:%M:%S", time.gmtime(end)))

        print(f"Previous best recall: {best_recall}")
        print(f"Recall: {t_test['recall']}")
        
        if args_sys.reversed == 1 and np.mean(t_test['recall']) > np.mean(best_recall):
            best_recall=t_test['recall']
            start = time.time()
            augmented_data = data_augment(model, dataset, args, args_sys, sess, args_sys.reversed_gen_number)
            end = time.time() - start
            print("Execution time for data augment: %s", time.strftime("%H:%M:%S", time.gmtime(end)))
            with open(aug_data_signature+'.txt', 'w') as f:
                for u, aug_ilist in augmented_data.items():
                    for ind, aug_i in enumerate(aug_ilist):
                        f.write(str(u-1) + '\t' + str(aug_i - 1) + '\t' + str(-(ind+1)) + '\n')
            if args_sys.reversed_gen_number > 0:
                if not os.path.exists(os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder)):
                    os.makedirs(os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder))
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, os.path.join(FIX_PATH,'reversed_models/'+args_sys.dataset+sub_reversed_folder+'/'+model_signature+'.ckpt'))
                print('Successed save model')
                
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            print(stat)

    except Exception as e:
        print("Exception during training and data augmentation:", e)
        traceback.print_exc()
    finally:
        if 'sampler' in locals():
            sampler.close()
        if 'sess' in locals():
            sess.close()
        print("Releasing resources")
        del model
        del sampler
        del dataset
        del args
        gc.collect()

    return np.mean(t_test['recall']) if 't_test' in locals() else np.mean(best_recall)

def run_trial(_):
    trial = study.ask()
    value = objective(trial)
    study.tell(trial, value)
    return value

if __name__ == '__main__':
    try:
        args_sys = parser.parse_args()
        max_num_core=10;spark_executor_memory=26; spark_driver_memory=26; offHeap_size=16
        check_available_memory(max_num_core, spark_driver_memory, spark_driver_memory)
        print("Starting Spark session...")
        spark = create_spark_session(max_num_core, spark_driver_memory, spark_driver_memory, offHeap_size)
        print("Starting Optuna optimization...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        trials_rdd = spark.sparkContext.parallelize(range(100), numSlices=100)
        
        start = time.time()
        # Execute trials
        results = trials_rdd.map(run_trial).collect()
        # Get the best hyperparameters
        best_trial = study.best_trial
        best_hps = best_trial.params
        print("Best RECALL: ", -best_trial.value)
        print("Best parameters: ", best_hps)
        # Save the best parameters to a dictionary
        json_best_hps = json.dumps(best_hps, indent=4)

        # Save the dictionary to a text file
        os.makedirs(os.path.join(FIX_PATH,'best_params.txt'), exist_ok = True)
        with open(os.path.join(FIX_PATH,'best_params.txt'), 'w') as f:
            for param, value in best_hps.items():
                f.write(f"{param}: {value}\n")
        print("Best parameters saved to {}".format(os.path.join(FIX_PATH,'best_params.txt')))
        end = time.time() - start
        print("Execution time for hyperparamter tuning: %s", time.strftime("%H:%M:%S", time.gmtime(end)))
    except ConnectionRefusedError as e:
        print(f"Connection refused: {e}")
    except tf.errors.ResourceExhaustedError:
        print('Resource exhausted. Try reducing the batch size.')
    except socket.gaierror as e:
        print(f"Address-related error occurred: {e}")
    except socket.error as e:
        print(f"Socket error occurred: {e}")
    except Exception as err:
        print(f"An error of type {type(err).__name__} occurred. Arguments:\n{str(err)}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                limit=2, file=sys.stdout)
    finally:
        spark.stop()

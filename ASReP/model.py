from modules import embedding, multihead_attention, normalize, feedforward
import tensorflow as tf

class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        tf.compat.v1.disable_eager_execution()
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        # self.is_training = True
        self.u = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.cast(tf.not_equal(self.input_seq, 0), dtype=tf.float32), -1)

        src_masks = tf.math.equal(self.input_seq, 0)

        with tf.compat.v1.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            # self.seq = tf.keras.layers.Dropout(
            #                              rate=args.dropout_rate,
            #                              training=self.is_training)(self.seq)
            self.seq = tf.keras.layers.Dropout(rate=args.dropout_rate)(self.seq)
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   values=self.seq,
                                                   key_masks=src_masks,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units])
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        #self.test_item = tf.placeholder(tf.int32, shape=(101))
        #test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        test_item_emb = item_emb_table
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum+1])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.compat.v1.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.compat.v1.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.cast(tf.not_equal(pos, 0), dtype=tf.float32), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.compat.v1.reduce_sum(
            - tf.math.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.math.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.compat.v1.reduce_sum(istarget)
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.compat.v1.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.compat.v1.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.compat.v1.summary.merge_all()

    def predict(self, sess, u, seq):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.is_training: False})

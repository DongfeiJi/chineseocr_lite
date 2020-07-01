import mxnet
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss, _apply_weighting


class MyBlock(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(MyBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2D(channels=2048,
                              kernel_size=1,
                              strides=1,
                              padding=0,
                              use_bias=False)
        self.pool = nn.GlobalAvgPool2D()
        self.flatten = nn.Flatten()

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class NewTripletLoss(Loss):
    def __init__(self, batch_size_per_gpu, margin=1, weight=None, batch_axis=0, **kwargs):
        super(NewTripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self.batch_size_per_gpu = batch_size_per_gpu
        self.margin = margin

    def hybrid_forward(self, F, embeddings, labels, sample_weight=None):
        N = self.batch_size_per_gpu
        # get distance
        xx = F.power(embeddings, 2).sum(1, keepdims=True).tile((1, self.batch_size_per_gpu))
        dist = F.broadcast_add(xx, xx.transpose())
        dist = F.broadcast_sub(dist, 2 * F.dot(embeddings, embeddings.transpose()))
        dist = F.clip(dist, 1e-12, 1e12).sqrt()
        print(dist)

        # get mask
        labels = F.cast(labels, dtype='float32')
        labels = labels.expand_dims(1).tile((1, self.batch_size_per_gpu))
        is_pos = F.broadcast_equal(labels, labels.transpose())
        is_neg = F.broadcast_not_equal(labels, labels.transpose())
        # hard example mining
        dist_mat = dist.reshape((self.batch_size_per_gpu * self.batch_size_per_gpu,))
        pos_mask = is_pos.reshape((self.batch_size_per_gpu * self.batch_size_per_gpu,))
        dist_ap = F.contrib.boolean_mask(dist_mat, pos_mask).reshape((self.batch_size_per_gpu, -1))
        # dist_ap = F.broadcast_mul(dist_mat, pos_mask).reshape((self.batch_size_per_gpu, -1))
        dist_ap = F.max(dist_ap, axis=1)
        neg_mask = is_neg.reshape((self.batch_size_per_gpu * self.batch_size_per_gpu,))
        dist_an = F.contrib.boolean_mask(dist_mat, neg_mask).reshape((self.batch_size_per_gpu, -1))
        # dist_an = F.broadcast_mul(dist_mat, neg_mask).reshape((self.batch_size_per_gpu, -1))
        dist_an = F.min(dist_an, axis=1)
        # add margin
        margin = F.full(shape=(self.batch_size_per_gpu, 1), val=self.margin)
        loss = F.broadcast_add(F.broadcast_sub(dist_ap, dist_an), margin)
        loss = F.maximum(loss, F.zeros_like(loss))
        # apply weight
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class Model(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.net = MyBlock()
        self.loss = NewTripletLoss(batch_size_per_gpu=64, margin=0.35)

    def hybrid_forward(self, F, x, y):
        x = self.net(x)
        loss = self.loss(x, y)
        return loss

if __name__ == "__main__":
    import numpy as np
    import random
    feat = np.random.rand(64, 3, 32, 100)
    feat = nd.array(feat)
    # feat = nd.random.randn(64, 2048)
    target = []
    label = [_ for _ in range(16)]
    for i in range(4):
        target += label
    random.shuffle(target)
    target = nd.array(target)
    model = Model()
    model.initialize()
    model.hybridize()
    loss = model(feat, target)
    print(loss)

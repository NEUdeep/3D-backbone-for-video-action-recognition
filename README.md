# 3D-backbone-for-video-action-recognition
3D backbone for video action recognition, the new model will come soon...


### 模型设计思想


1）C3D
从上面的模型可以看到，我对于输入和输出，以及中间层进行了一些优化，但是依然存在性能考量的不断思考。C3D我原先实现的Version里没有考虑到output的shape，导致实际中存在问题，当前更新版本已经更新。我的希望是pooling层能够对clip以及map的dim进行完美的处理，达到性能不降低的网络设计。首先考量clip的dim对于时空关系modeling的重要性，我们在strider上进行了精确计算，在最后一层一直保持dim不变。针对map的dim我们也一样进行了处理，之前实现的C3D存在3层fc，期望信息线性变化的将video信息能够一边降维的同时，整合学到的空间特征，以及时空特征信息，但是，我们的3层fc的计算量是非常的巨大的，既然pooling层之前发挥了应有的作用，所以就直接将map到(1，1)在pooling部分处理，但是我们也清楚这样子的跃进会带来特征信息的跳跃性变化，减弱channel之间的关联关系，但是当前与fc带来的计算量相比，这可以忍受。

1）MFNet
无独有偶，P3D，MFNet等设计当中，也偶合我关于C3D上面的考量。但是，3D模型结构依然存在着很多问题，尤其被应用到video的任务当中，或者3d(医疗data)当中，还不是很有效。这一块的研究也由于应用领域问题的巨大红利，激发这我们青年一代不断探索前进......

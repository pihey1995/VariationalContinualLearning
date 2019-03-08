import numpy as np
import discriminative.utils.vcl as vcl
import discriminative.utils.coreset as coreset
from discriminative.utils.DataGenerator import PermutedMnistGenerator

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

np.random.seed(1)
#Just VCL
coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
np.save("./results/VCL{}".format(""), vcl_result)
print(vcl_result)

#VCL + Random Coreset
np.random.seed(1)

for coreset_size in [200,400,1000,2500,5000]:
    data_gen = PermutedMnistGenerator(num_tasks)
    rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, single_head, gan_bol=True)
    np.save("./results/rand-VCL-{}".format(coreset_size), rand_vcl_result)
    print(rand_vcl_result)

#VCL + k-center coreset
np.random.seed(1)
coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)
np.save("./results/kcen-VCL{}".format(coreset_size), kcen_vcl_result)


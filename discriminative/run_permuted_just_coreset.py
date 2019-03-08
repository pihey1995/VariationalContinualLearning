import numpy as np
import discriminative.utils.vcl as vcl
import discriminative.utils.coreset as coreset
from discriminative.utils.DataGenerator import PermutedMnistGenerator

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

np.random.seed(0)
#for coreset_size in [400,1000,2500,5000]:
#    data_gen = PermutedMnistGenerator(num_tasks)
#    vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
#        coreset.rand_from_batch, coreset_size, batch_size, single_head)
#    np.save("./results/only-coreset-{}".format(coreset_size), vcl_result)
#    print(vcl_result)

np.random.seed(0)
coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = vcl.run_coreset_only(hidden_size, no_epochs, data_gen,
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)
np.save("./results/kcen-coreset-only{}".format(coreset_size), kcen_vcl_result)



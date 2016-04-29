import numpy as np

caffe_root = './'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe

def get_caffe_iter(layer_names, layers):
  for layer_idx, layer in enumerate(layers):
    layer_name = layer_names[layer_idx].replace('/', '_')
    layer_type = layer.type
    layer_blobs = layer.blobs
    yield (layer_name, layer_type, layer_blobs)

layers = ''
layer_names = ''

caffe.set_mode_cpu()
net_caffe = caffe.Net('./examples/mnist/deploy.prototxt', './examples/mnist/models/lenet_iter_10000.caffemodel', caffe.TEST)
layer_names = net_caffe._layer_names
layers = net_caffe.layers

iter = ''
iter = get_caffe_iter(layer_names, layers)

fout = open('handPose.txt', 'wr')

first_conv = True
for layer_name, layer_type, layer_blobs in iter:
  print (layer_name, layer_type, layer_blobs)
  if layer_type == 'Convolution' or layer_type == 'InnerProduct' or layer_type == 4 or layer_type == 14:
    assert(len(layer_blobs) == 2)
    wmat = np.array(layer_blobs[0].data)#reshape(layer_blobs[0].num, layer_blobs[0].channels, layer_blobs[0].height, layer_blobs[0].width)
    bias = np.array(layer_blobs[1].data)
    print wmat.shape
    print bias.shape

    for k in range(len(wmat.shape)):
      fout.write("{} ".format(wmat.shape[k]))
    fout.write("\n{}\n".format(bias.shape[0]))
    
    if layer_type == 'Convolution':
      for i in range(wmat.shape[0]):
        for j in range(wmat.shape[1]):
          np.savetxt(fout, wmat[i, j, :, :])
    if layer_type == 'InnerProduct':
      np.savetxt(fout, wmat)
    np.savetxt(fout, bias)

fout.close()

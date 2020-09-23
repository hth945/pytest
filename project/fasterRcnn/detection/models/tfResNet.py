import tensorflow as tf
from tensorflow.keras import layers

def getResNetRpn(scal=224):

    sampleModel = tf.keras.applications.ResNet50V2(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=(scal, scal, 3))
    # sampleModel.trianable = False
    c = []
    name = ['conv2_block2_out', 'conv3_block3_out', 'conv4_block5_out', 'conv5_block3_out']
    i = 0
    for l in sampleModel.layers:
        if l.name == name[i]:
            i += 1
            # print(l.name)
            c.append(l.output)
            if i == 4:
                break
    # print(c)
    model = tf.keras.models.Model(inputs=sampleModel.input, outputs=c)
    # tf.keras.utils.plot_model(model, to_file='rennetRpn.png', show_shapes=True, show_layer_names=True)
    return model


# %%
if __name__ == '__main__':
    scal = 512#768
    random_float = tf.random.uniform(shape=(10, scal, scal, 3))
    backbone = getResNetRpn(scal)

    C2, C3, C4, C5 = backbone(random_float)
    print(C2.shape)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)
    print(backbone.compute_output_shape((1, scal, scal, 3)))
    # print(backbone.compute_output_shape((1, 768, 768, 3)))

# %%

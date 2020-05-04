import tensorflow as tf

import numpy as np
import cv2

tf.reset_default_graph()                                                 # graph 초기화
saver = tf.train.import_meta_graph('../lip_dont_use/trained/model/model-711600.meta')      # network 불러오기

sess = tf.Session()
# 아래 두개중 하나를 사용하시면 됩니다.
saver.restore(sess, tf.train.latest_checkpoint('../lip_dont_use/trained/model/'))                    # 가장 마지막 checkpoint
# saver.restore(sess, './my_model/my_model-1000')                          # 지정한 이름의 checkpoint

graph = tf.Graph()
with graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(frozen_graph_filename, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

image_tensor = graph.get_tensor_by_name('image_tensor:0')
output_tensor = graph.get_tensor_by_name('generate_output/output:0')
sess = tf.compat.v1.Session(graph=graph)

generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
print(sess.run('weight:0'))



def load_examples():
    input_path = '../lip_dont_use/trained/input/34.jpg'
    decode = tf.image.decode_jpeg

    raw_input = decode(input_path)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

    with tf.name_scope("load_images"):
        # path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        # reader = tf.WholeFileReader()
        # paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

examples = load_examples()

# if a.lab_colorization:
outputs = augment(model.outputs, examples.inputs)
inputs = deprocess(examples.inputs)
# else:
#     inputs = deprocess(examples.inputs)
#     # targets = deprocess(examples.targets)
#     outputs = deprocess(model.outputs)

def convert(image):
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

with tf.name_scope("convert_inputs"):
    converted_inputs = convert(inputs)

# with tf.name_scope("convert_targets"):
#     converted_targets = convert(targets)

with tf.name_scope("convert_outputs"):
    converted_outputs = convert(outputs)

display_fetches = {
    "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
    # "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
    "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
}

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

sv = tf.train.Supervisor(logdir='../lip_dont_use/trained/output', save_summaries_secs=0, saver=None)
with sv.managed_session() as sess:
    results = sess.run(display_fetches)

    # filesets = save_images(results)
import tensorflow as tf

# 加载 CIFAR-100 数据集
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# 创建 ResNet50 模型
model = tf.keras.applications.ResNet50(
include_top=True,
weights=None,
input_shape=(32, 32, 3),
classes=100,)

# 编译模型
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=64)


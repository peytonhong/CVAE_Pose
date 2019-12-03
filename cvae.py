import tensorflow as tf
import numpy as np

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(112, 112, 1)),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
        )

        self.generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(2, 2), padding="SAME"),
            ]
        )

        self.pose_net = tf.keras.Sequential(
            [
            # tf.keras.layers.InputLayer(input_shape=(latent_dim*2,)),
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=100, activation='tanh'),
            tf.keras.layers.Dense(units=1)
            ]
        )


    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits



def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), # input z 값이 random sampling 되어 log를 취한 Gaussian distribution의 PDF 값을 계산하는 식
        axis=raxis)

@tf.function
def compute_loss(model, x, y, pose_gt):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
    
    # pose_est = model.pose_net(tf.reshape(tf.stack([mean, logvar], axis=1), (-1, model.latent_dim*2)))
    pose_est = model.pose_net(z)
    pose_loss = tf.keras.losses.MSE(tf.reshape(pose_est, (-1,)), pose_gt)
    
    # ELBO를 계산하는 방법 중 Jensen's Inequality 방식을 적용한 수식: Monte Carlo estimator 방식 사용 (for simplicity)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)                       # KL divergence term: D_KL = E[log q(z|x)] - E[log p(z)]로부터 나온 식
    logqz_x = log_normal_pdf(z, mean, logvar)               # KL divergence term: D_KL = E[log q(z|x)] - E[log p(z)]로부터 나온 식
    return -tf.reduce_mean(logpx_z + logpz - logqz_x) + pose_loss, mean, pose_est

@tf.function
def compute_apply_gradients(model, x, y, pose_gt, optimizer):
    with tf.GradientTape() as tape:
        loss, _, _ = compute_loss(model, x, y, pose_gt)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
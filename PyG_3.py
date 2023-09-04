# PyTorch Implementation (CPU)
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a PyTorch model
model = Net()

# Generate some random data
batch_size = 64
num_batches = 100
input_data = torch.randn(batch_size * num_batches, 1, 28, 28)
target_data = torch.randint(0, 10, (batch_size * num_batches,))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
start_time = time.time()
for epoch in range(5):
    running_loss = 0.0
    for i in range(num_batches):
        inputs = input_data[i * batch_size: (i + 1) * batch_size]
        labels = target_data[i * batch_size: (i + 1) * batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / num_batches:.4f}")

end_time = time.time()
print(f"PyTorch Training Time: {end_time - start_time:.2f} seconds")

# JAX Implementation on TPU
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax

# Define a simple neural network using Flax
class Net(nn.Module):
    features: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.features)
        self.dense2 = nn.Dense(features=10)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x

# Create a JAX model
key = jax.random.PRNGKey(0)
input_shape = (batch_size, 1, 28, 28)
model_def = Net(features=512)
params = model_def.init(key, jnp.ones(input_shape))

# Define training loss and optimizer
def loss_fn(params, images, targets):
    logits = model_def.apply(params, images)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))
    return loss

opt = optax.sgd(learning_rate=0.01, momentum=0.9)
opt_state = opt.init(params)

# JIT-compile the training step
@jax.jit
def train_step(params, opt_state, images, targets):
    grads = jax.grad(loss_fn)(params, images, targets)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Generate random data
input_data = jnp.array(input_data)
target_data = jnp.array(target_data)

# Training loop
num_epochs = 5
start_time = time.time()
for epoch in range(num_epochs):
    for i in range(num_batches):
        images = input_data[i * batch_size: (i + 1) * batch_size]
        labels = target_data[i * batch_size: (i + 1) * batch_size]

        params, opt_state = train_step(params, opt_state, images, labels)

    loss = loss_fn(params, images, labels)
    print(f"Epoch {epoch + 1}, Loss: {loss}")

end_time = time.time()
print(f"JAX (on TPU) Training Time: {end_time - start_time:.2f} seconds")

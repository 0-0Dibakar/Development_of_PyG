# Development_of_PyG

                                     ![img2](https://github.com/0-0Dibakar/Development_of_PyG/assets/106139442/627f79c1-626b-46b3-b780-7189617f4a41)

I developed an PyG is a pure-Python library for creating and modifying graph neural networks (GNNs), built on top of PyTorch and extend PyG such that it is compatible with all other ML frameworks (TensorFlow, JAX, NumPy, etc.)
PyG Multi-Framework Compatibility Project
PyG is a powerful library for creating and modifying graph neural networks (GNNs) using PyTorch. However, this project aims to take PyG to the next level by making it compatible with multiple machine learning frameworks, such as TensorFlow, JAX, NumPy, and more. This compatibility will be achieved by leveraging Ivy's transpiler in the backend.

Project Overview
The primary goals of this project are as follows:

Framework Handler Implementation: Developing a framework handler for PyG that will allow seamless multi-framework support. This handler will enable users to switch between different machine learning frameworks (PyTorch, TensorFlow, JAX, NumPy, etc.) with ease, providing a consistent interface for graph neural network operations.

![IMG1](https://github.com/0-0Dibakar/Development_of_PyG/assets/106139442/c2ff2381-e916-437d-aeda-5f73b4d34889)


Ivy Integration: Implementing missing functions in Ivy's backend and frontend so that all functions and operations in PyG can be correctly transpiled. Ivy will play a crucial role in translating PyG's operations into the syntax and functionality of various ML frameworks.

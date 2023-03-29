import numpy as np

nodes = [f"M{n}" for n in range(3000)]
nodes_kernel1 = {node: i for i, node in enumerate(np.random.choice(nodes, 1300, replace=False))}
nodes_kernel2 = {node: i for i, node in enumerate(np.random.choice(nodes, 400, replace=False))}
nodes_kernel3 = {node: i for i, node in enumerate(np.random.choice(nodes, 150, replace=False))}
big_kernel1 = np.random.randint(0, 10, (len(nodes_kernel1), len(nodes_kernel1)))
big_kernel2 = np.random.randint(0, 10, (len(nodes_kernel2), len(nodes_kernel2)))
big_kernel3 = np.random.randint(0, 10, (len(nodes_kernel3), len(nodes_kernel3)))

np.save("./bigkernel1.npy", big_kernel1)
np.save("./bigkernel2.npy", big_kernel2)
np.save("./bigkernel3.npy", big_kernel3)

for idx, nodes_kernel in enumerate([nodes_kernel1, nodes_kernel2, nodes_kernel3]):
    with open(f"./bigkernel{idx+1}.lst", "w") as f:
        for node in nodes_kernel.keys():
            f.write(node)
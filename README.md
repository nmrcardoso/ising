# Ising

The code supports:
- 1D, 2D and 3D
- CPU with OpenMP
    - implicitly run when alocationg the lattice aray with "Host"
		- Array<int> *lattice = new Array<int>(Host, Volume());
- NVIDIA GPUs with CUDA
    - implicitly run when alocationg the lattice aray with "Device"
		- Array<int> *lattice = new Array<int>(Device, Volume());
- the results are plot in real time using gnuplot

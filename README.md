# Ising

The code supports:
- CPU with OpenMP
    - implicitly run when alocationg the lattice aray with "Host"
		- Array<int> *lattice = new Array<int>(Host, Volume()*Dirs());
- NVIDIA GPUs with CUDA
    - implicitly run when alocationg the lattice aray with "Device"
		- Array<int> *lattice = new Array<int>(Device, Volume()*Dirs());
- the results are plot in real time using gnuplot

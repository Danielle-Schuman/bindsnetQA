Based on BindsNET – a spiking neural network simulation library geared towards the development of biologically inspired algorithms for machine learning – this Bachelor-Project tries to replace the forward-step of the "supervised_mnist"-example (in network.run(...) in bindsnet_qa/network/network.py) with the usage of a D-Wave Quantum Annealer or a simulator thereof using D-Waves qbsolv Package.

Documentation for the BindsNET-package can be found [here](https://bindsnet-docs.readthedocs.io).

## Requirements

- Python 3.6
- `requirements.txt`

## Setting things up

### Using pip
To build the `bindsnet_qa` package from source, clone the GitHub repository, change directory to the top level of this project, and issue

```
pip install .
```

Or, to install in editable mode (allows modification of package without re-installing):

```
pip install -e .
```

## Getting started

To run a near-replication of the SNN from [this paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full), issue

```
cd examples/mnist
python supervised_mnist.py --time 5 --update_interval 10 --n_train 250 --n_neurons 10
```

Caveat: Runs a little slow.

There are a number of optional command-line arguments which can be passed in, including `--plot` (displays useful monitoring figures), `--time [int]` (determines the number of forward-timesteps per MNIST-Datum),  `--n_train [int]` (total number of training iterations), `--update_interval [int]` (determines how often the current accuracy is shown), and more. 
Run the script with the `--help` or `-h` flag for more information.

If you want to make plots and save them to a certain directory, use the arguments `--plot -- directory [the path to the directory you want to save them in]`.

## Running the tests

Issue the following to run the tests:

```
python -m pytest test/
```

Some tests will fail if Open AI `gym` is not installed on your machine.

## Background
TODO

## Benchmarking
As of now, it runs slower compared to the original BindsNET-version. Reasons for this are being investigated.

## Citation

As I am using BindsNET, I'm hereby citing [article](https://www.frontiersin.org/article/10.3389/fninf.2018.00089):

```
@ARTICLE{10.3389/fninf.2018.00089,
	AUTHOR={Hazan, Hananel and Saunders, Daniel J. and Khan, Hassaan and Patel, Devdhar and Sanghavi, Darpan T. and Siegelmann, Hava T. and Kozma, Robert},   
	TITLE={BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python},      
	JOURNAL={Frontiers in Neuroinformatics},      
	VOLUME={12},      
	PAGES={89},     
	YEAR={2018}, 
	URL={https://www.frontiersin.org/article/10.3389/fninf.2018.00089},       
	DOI={10.3389/fninf.2018.00089},      
	ISSN={1662-5196},
}

```

## Contributors
- Daniëlle Schuman

To BindsNET:
- Daniel Saunders ([email](mailto:djsaunde@cs.umass.edu))
- Hananel Hazan ([email](mailto:hananel@hazan.org.il))
- Darpan Sanghavi ([email](mailto:dsanghavi@cs.umass.edu))
- Hassaan Khan ([email](mailto:hqkhan@umass.edu))
- Devdhar Patel ([email](mailto:devdharpatel@cs.umass.edu))



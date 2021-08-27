# Cart Pole Optimisation

## About
Investigation into different methods of controlling the cart-pole experiment.

## Structure
Each file contains a different method:
* `basic.py` - no control applied.
* `pid_hillclimb.py` - PID controller with values optimised via hill climb method.
* `pid_iterative.py` - PID controller with values obtained via iteration.
* `qlearning.py` - Q-Learning algorithm.

## Installation
To run the code you will need to have a version of Python 3.x running on your local machine. Download this repository and then run `pip install -r requirements.txt` in a command line to collect the third party libraries. Then each method can be run individually by `py {METHOD_NAME}.py`.

# Usage
During the solve, current progress will be logged to the command line. After each method has been run, a series of charts will be automatically displayed which show the performance.

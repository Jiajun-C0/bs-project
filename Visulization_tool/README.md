# Bacon-Shor Code Visualization

## Overview
This Python code implements a visualization for the Bacon-Shor code. It provides an interactive grid where users can select qubit pairs for measurement, and visualizes the associated stabilizers and measurements.

## Code Files Overview

### 1. `BaconShorViz_Initial.py`
This file contains the initial implementation of the Bacon-Shor code visualization. Limited to text input. 

### 2. `BaconShorViz_Interactive.py`
This file enhances the initial implementation by adding interactive functionality. 



## Features
- **Interactive Grid Visualization**: Users can click on edges of the grid to select measurements.
- **Stabilizer Visualization**: The grid highlights stabilizers based on the types of operations ('X' or 'Z') involved.
- **Measurement Representation**:
  - **Solid Lines**: Represent current measurements.
  - **Dashed Lines**: Represent previous measurements.
- **Grid Cells**: Each cell in the grid is shaded based on the stabilizers it contains.

## Dependencies
- `numpy`
- `matplotlib`

```bash pip install -r requirements.txt```

## Code Structure
### Classes
1. **Stabilizer**: Represents a stabilizer with its qubit operations and methods to check commuting and anti-commuting relations.
2. **BaconShorCode**: Represents the Bacon-Shor code and manages the qubit grid, measurements, and stabilizers.

### Main Functions
- **run_time_step**: Advances the time step and processes measurements.
- **draw_grid**: Visualizes the qubit grid, measurements, and stabilizers.
- **start_interactive_session**: Initiates an interactive session for selecting measurements.

## Visualizations
The visualization includes several key components:

### Qubit Grid
- The qubit grid is represented as a series of points (vertices) arranged in rows and columns. Each point corresponds to a qubit.

### Measurements
- **Solid Lines**: Represent current measurements. They connect two qubits that are being measured together.
- **Dashed Lines**: Represent previous measurements. They indicate measurements made in earlier time steps.

### Stabilizers
- Stabilizers are visualized by shading the cells in the grid:
  - **Light Blue**: Indicates stabilizers with 'X' operators.
  - **Light Pink**: Indicates stabilizers with 'Z' operators.

### Example Visualization
The visualization is dynamically generated as users select measurements. The following provides an example of how the grid might look:
```
Qubit Grid Positions:
  1  2  3  4  5 
  6  7  8  9 10 
 11 12 13 14 15 
 16 17 18 19 20 
 21 22 23 24 25 
```

### Running the Code
To run the code, execute the following command:
```bash
python <your_script_name.py>
```
This will open an interactive session where you can select edges for measurements. 

### Exiting the Code

Please note that you need to exit the code manually after you’re done interacting with the visualization. If you do not exit, the code will continue to run, allowing for continuous interaction with the grid.


## Usage
1. Run the script to open the interactive visualization.
2. Click on the edges of the grid to select pairs of qubits for measurement.
3. Close the visualization window to process the selected measurements and visualize the updated state.

## Known Issues (vision 0.1.0)
- **Stabilizer generator visualization**: Stabilizers may not be displayed correctly if a stabilizer generator is broken by a measurement in the middle. This will be addressed in future versions.

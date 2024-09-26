import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class Stabilizer:
    def __init__(self, qubits_ops):
        self.qubits_ops = qubits_ops  # {qubit_index: 'X' or 'Z'}

    def commutes_with(self, measurement):
        '''
        Check the satbilziers if they commute with the pair measurement.
        if a stabilizer has even number of anti-commute qubits with the pair measurement, it is commute with the pair measurement
        '''
        anti_commuting_qubits = 0
        for q in measurement['qubits']: # Qubits used in the pair measurements eg. measurement = {'qubits': [1, 2], 'type': 'X'}
            # self.qubits_ops = {1: 'X', 3: 'Z'} is a dictionary where keys are qubit indices (e.g., 1, 2, 3, etc.), 
            # and values are operators applied to those qubits (e.g., 'X', 'Z', etc.).
            # Checking if qubit q (involved in the measurement) has an operator assigned to it in the self.qubits_ops dictionary. 
            # If the qubit doesn’t have an operator assigned, it assumes the qubit is in the identity state 'I'
            op_stab = self.qubits_ops.get(q, 'I') # Operator on qubit q from stabilizer (defaults to identity 'I')
            op_meas = measurement['type'] # Measurement type (e.g., 'X', 'Z')
            
            # If the stabilizer has a non-identity operator ('X', 'Z') and it doesn't match the measurement type,
            # it anti-commutes. We count these anti-commuting cases.
            if op_stab != 'I' and op_stab != op_meas:
                anti_commuting_qubits += 1
        return anti_commuting_qubits % 2 == 0 

    def multiply(self, other):
        ''' 
        other: the other stabilizer we want to combine with
        '''
        new_qubits_ops = self.qubits_ops.copy()
        for q, op in other.qubits_ops.items():
            if q in new_qubits_ops:
                if new_qubits_ops[q] == op:
                    #this qubit is removed from new_qubits_ops
                    del new_qubits_ops[q]  # X*X = I, Z*Z = I
                else:
                    new_qubits_ops[q] = 'Y'  # X*Z or Z*X = Y
            else:
                new_qubits_ops[q] = op
        return Stabilizer(new_qubits_ops)

    def is_trivial(self):
        return len(self.qubits_ops) == 0

    def __eq__(self, other):
        return self.qubits_ops == other.qubits_ops

    def __hash__(self):
        return hash(frozenset(self.qubits_ops.items()))

    def __str__(self):
        return ' '.join([f"{op}{q}" for q, op in sorted(self.qubits_ops.items())])

class BaconShorCode:
    def __init__(self, rows, cols):
        self.rows = rows  # Number of cells vertically
        self.cols = cols  # Number of cells horizontally

        # Defining a grid of qubits with given rows and columns
        self.num_qubits = (rows + 1) * (cols + 1)
        self.grid = np.arange(1, self.num_qubits + 1).reshape((rows + 1, cols + 1))
        self.stabilizer_generators = set()
        self.time_step = 0
        self.prev_measurements = []
        self.current_measurements = []
        self.measurements_history = []
        self.selected_edges = []
        self.print_qubit_grid()

    def print_qubit_grid(self):
        '''
        Example Output for 5x5 grid: 
        Qubit Grid Positions:
            1  2  3  4  5 
            6  7  8  9 10 
            11 12 13 14 15 
            16 17 18 19 20 
            21 22 23 24 25 
        '''
        print("Qubit Grid Positions:")
        for i in range(self.rows + 1):
            row = ''
            for j in range(self.cols + 1):
                q = self.grid[i, j]
                row += f'{q:2d} '  #take up to 2 character spaces + type dicimal
            print(row)
        print("\nQubit positions are given by (row, column) indices starting from (0,0).")
        # print("Use these indices to determine adjacent qubits for measurements.\n")

    def is_valid_measurement(self, qubit_pair):
        q1, q2 = qubit_pair #list with 2 entries eg.[1,2]

        # Find the coordinate of the qubit in the grid. output array([[0,1]]), where [0,1] is the coordinate of the qubite in the grid
        pos1 = np.argwhere(self.grid == q1) 
        pos2 = np.argwhere(self.grid == q2)
        if pos1.size == 0 or pos2.size == 0:
            return False, "Qubit does not exist in the grid."
        row_diff = abs(pos1[0][0] - pos2[0][0])
        col_diff = abs(pos1[0][1] - pos2[0][1])
        if (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1):
            return True, ""
        else:
            return False, "Qubits are not adjacent."

    def get_measurement_type(self, qubit_pair):
        q1, q2 = qubit_pair
        pos1 = np.argwhere(self.grid == q1)[0]
        pos2 = np.argwhere(self.grid == q2)[0]
        if abs(pos1[0] - pos2[0]) == 1 and pos1[1] == pos2[1]:
            return 'X'  # Vertical neighbor (XX measurement)
        elif abs(pos1[1] - pos2[1]) == 1 and pos1[0] == pos2[0]:
            return 'Z'  # Horizontal neighbor (ZZ measurement)
        else:
            return None

    def run_time_step(self, measurements):
        self.time_step += 1
        print(f"\n=== Time Step {self.time_step} ===")
        valid_measurements = []
        for qubit_pair in measurements:
            valid, msg = self.is_valid_measurement(qubit_pair)
            if not valid:
                print(f"Invalid measurement {qubit_pair}: {msg}")
            else:
                valid_measurements.append(qubit_pair)
        self.update_stabilizers(valid_measurements)
        # Store current measurements
        self.current_measurements = valid_measurements
        self.measurements_history.append(valid_measurements)
        self.draw_grid()
        # Clear temporary measurement lists
        self.selected_edges = []

    def simplify_stabilizers(self):
        # Sort stabilizers by weight (number of qubits), ascending
        sorted_stabilizers = sorted(self.stabilizer_generators, key=lambda s: len(s.qubits_ops))
        simplified_stabilizers = set()
        for stab in sorted_stabilizers:
            # Remove subsets
            for simple_stab in simplified_stabilizers:
                # If simple_stab is a subset of stab
                if set(simple_stab.qubits_ops.items()).issubset(set(stab.qubits_ops.items())):
                    # Remove simple_stab from stab by multiplying
                    stab = stab.multiply(simple_stab)
            if not stab.is_trivial():
                simplified_stabilizers.add(stab)
        # Update the stabilizer generators
        self.stabilizer_generators = simplified_stabilizers

    def update_stabilizers(self, measurements):
        new_stabilizers = set()
        measurement_stabilizers = []

        for meas in measurements:
            meas_type = self.get_measurement_type(meas)
            qubits_ops = {q: meas_type for q in meas}
            meas_stab = Stabilizer(qubits_ops)
            measurement_stabilizers.append(meas_stab) 
            new_stabilizers.add(meas_stab)
        
        if self.time_step == 1:
            # Initialise the stabilizer generator group
            self.stabilizer_generators = new_stabilizers
        else:
            # Adds all elements from new_stabilizers to self.stabilizer_generators without duplicates
            self.stabilizer_generators |= new_stabilizers
            self.adjust_stabilizers(measurement_stabilizers)
            # Simplify stabilizers after adjustment
            self.simplify_stabilizers()
        # Print stabilizers for debugging
        print("Stabilizer Generators:")
        for stab in self.stabilizer_generators:
            print(str(stab))
    
    def get_qubit_position(self, qubit):
        pos = np.argwhere(self.grid == qubit)
        if pos.size == 0:
            raise ValueError(f"Qubit {qubit} not found in the grid.")
        return pos[0]  # Returns (row_index, column_index)

    def adjust_stabilizers(self, measurement_stabilizers):
        for meas_stab in measurement_stabilizers:
            # Finds all anti-commuting stabilizers in self.stabilizer_generators with the measurement
            anti_commuting_stabs = [stab for stab in self.stabilizer_generators if not 
                                    stab.commutes_with({'qubits': list(meas_stab.qubits_ops.keys()), 
                                                        'type': list(meas_stab.qubits_ops.values())[0]}) 
                                    and stab != meas_stab]
            changed = True
            while changed:
                changed = False
                # Make a copy to iterate over
                anti_commuting_stabs_list = anti_commuting_stabs.copy()
                for i, stab1 in enumerate(anti_commuting_stabs_list):
                    stab_type = list(stab1.qubits_ops.values())[0]  # 'X' or 'Z'
                    for stab2 in anti_commuting_stabs_list[i+1:]:
                        if stab_type == list(stab2.qubits_ops.values())[0]:  # Same type
                            if stab_type == 'Z':
                                cols1 = {self.get_qubit_position(q)[1] for q in stab1.qubits_ops}
                                cols2 = {self.get_qubit_position(q)[1] for q in stab2.qubits_ops}
                                if len(cols1.union(cols2)) <= 2:
                                    # Combine stabilizers
                                    self.stabilizer_generators.discard(stab1)
                                    self.stabilizer_generators.discard(stab2)
                                    anti_commuting_stabs.remove(stab1)
                                    anti_commuting_stabs.remove(stab2)
                                    combined_stab = stab1.multiply(stab2)
                                    if not combined_stab.is_trivial():
                                        self.stabilizer_generators.add(combined_stab)
                                        # If combined stabilizer still anti-commutes, add it back
                                        if not combined_stab.commutes_with({'qubits': list(meas_stab.qubits_ops.keys()), 
                                                                            'type': list(meas_stab.qubits_ops.values())[0]}):
                                            anti_commuting_stabs.append(combined_stab)
                                    changed = True
                                    break  # Break inner loop to restart
                            elif stab_type == 'X':
                                rows1 = {self.get_qubit_position(q)[0] for q in stab1.qubits_ops}
                                rows2 = {self.get_qubit_position(q)[0] for q in stab2.qubits_ops}
                                if len(rows1.union(rows2)) <= 2:
                                    # Combine stabilizers
                                    self.stabilizer_generators.discard(stab1)
                                    self.stabilizer_generators.discard(stab2)
                                    anti_commuting_stabs.remove(stab1)
                                    anti_commuting_stabs.remove(stab2)
                                    combined_stab = stab1.multiply(stab2)
                                    if not combined_stab.is_trivial():
                                        self.stabilizer_generators.add(combined_stab)
                                        # If combined stabilizer still anti-commutes, add it back
                                        if not combined_stab.commutes_with({'qubits': list(meas_stab.qubits_ops.keys()), 
                                                                            'type': list(meas_stab.qubits_ops.values())[0]}):
                                            anti_commuting_stabs.append(combined_stab)
                                    changed = True
                                    break  # Break inner loop to restart
                    if changed:
                        break  # Break outer loop to restart
            # After attempting combinations, remove any remaining anti-commuting stabilizers
            for stab in anti_commuting_stabs:
                self.stabilizer_generators.discard(stab)

    def draw_grid(self):     
        fig, ax = plt.subplots(figsize=(self.cols + 1, self.rows + 1))
        ax.set_xlim(-0.5, self.cols + 0.5)
        ax.set_ylim(-0.5, self.rows + 0.5)
        ax.set_aspect('equal')
        # Draw grid lines (edges)
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                if j < self.cols:
                    ax.plot([j, j+1], [self.rows - i, self.rows - i], color='gray', linewidth=0.5)
                if i < self.rows:
                    ax.plot([j, j], [self.rows - i, self.rows - i - 1], color='gray', linewidth=0.5)
        # Draw qubits at vertices
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                q = self.grid[i, j]
                ax.plot(j, self.rows - i, 'o', color='black')
                ax.text(j + 0.1, self.rows - i + 0.1, str(q), ha='left', va='bottom', fontsize=9, color='black')
        # Draw previous measurements with dashed lines
        for qubit_pair in self.prev_measurements:
            q1, q2 = qubit_pair
            meas_type = self.get_measurement_type(qubit_pair)
            pos1 = np.argwhere(self.grid == q1)[0]
            pos2 = np.argwhere(self.grid == q2)[0]
            x1, y1 = pos1[1], self.rows - pos1[0]
            x2, y2 = pos2[1], self.rows - pos2[0]
            color = 'blue' if meas_type == 'X' else 'red'
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, linestyle='--', alpha=0.7)
        # Draw current measurements with solid lines
        for qubit_pair in self.current_measurements:
            q1, q2 = qubit_pair
            meas_type = self.get_measurement_type(qubit_pair)
            pos1 = np.argwhere(self.grid == q1)[0]
            pos2 = np.argwhere(self.grid == q2)[0]
            x1, y1 = pos1[1], self.rows - pos1[0]
            x2, y2 = pos2[1], self.rows - pos2[0]
            color = 'blue' if meas_type == 'X' else 'red'
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        # Draw stabilizer generators
        for stab in self.stabilizer_generators:
            # Determine color based on stabilizer type
            ops = set(stab.qubits_ops.values())
            if ops == {'X'}:
                color = 'lightblue'
            elif ops == {'Z'}:
                color = 'lightpink'
            else:
                color = 'lightgreen'  # For mixed operators
            # Collect qubits in stabilizer
            stab_qubits = set(stab.qubits_ops.keys())
            # Iterate over all cells
            for i in range(self.rows):
                for j in range(self.cols):
                    # Get qubits at the corners of the cell
                    q_tl = self.grid[i, j]
                    q_tr = self.grid[i, j+1]
                    q_bl = self.grid[i+1, j]
                    q_br = self.grid[i+1, j+1]
                    cell_qubits = {q_tl, q_tr, q_bl, q_br}
                    # Check if all qubits in the cell are in the stabilizer
                    if cell_qubits.issubset(stab_qubits):
                        # Shade the cell
                        x, y = j, self.rows - i - 1
                        rect = Rectangle((x, y), 1, 1, linewidth=1, edgecolor='none', facecolor=color, alpha=0.5)
                        ax.add_patch(rect)
        ax.axis('off')
        ax.set_title(f"Time Step {self.time_step}")
        plt.show()
    

    def normalize_edge(self, edge):
        return tuple(sorted(edge))

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Find the closest edge to the click
        x_click, y_click = event.xdata, event.ydata
        min_dist = float('inf')
        closest_edge = None
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                q1 = self.grid[i, j]
                pos1 = np.array([j, self.rows - i])
                # Horizontal edges
                if j < self.cols:
                    q2 = self.grid[i, j+1]
                    pos2 = np.array([j+1, self.rows - i])
                    edge_mid = (pos1 + pos2) / 2
                    dist = np.hypot(edge_mid[0] - x_click, edge_mid[1] - y_click)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = (q1, q2)
                # Vertical edges
                if i < self.rows:
                    q2 = self.grid[i+1, j]
                    pos2 = np.array([j, self.rows - (i+1)])
                    edge_mid = (pos1 + pos2) / 2
                    dist = np.hypot(edge_mid[0] - x_click, edge_mid[1] - y_click)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = (q1, q2)
        if closest_edge:
            norm_edge = self.normalize_edge(closest_edge)
            if norm_edge in self.selected_edges:
                # Edge already selected, remove it
                self.selected_edges.remove(norm_edge)
            else:
                self.selected_edges.append(norm_edge)
            self.update_interactive_plot()

    def update_interactive_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.cols + 0.5)
        self.ax.set_ylim(-0.5, self.rows + 0.5)
        self.ax.set_aspect('equal')
        # Draw grid lines and qubits
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                if j < self.cols:
                    self.ax.plot([j, j+1], [self.rows - i, self.rows - i], color='gray', linewidth=0.5)
                if i < self.rows:
                    self.ax.plot([j, j], [self.rows - i, self.rows - i - 1], color='gray', linewidth=0.5)
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                q = self.grid[i, j]
                self.ax.plot(j, self.rows - i, 'o', color='black')
                self.ax.text(j + 0.1, self.rows - i + 0.1, str(q), ha='left', va='bottom', fontsize=9, color='black')
        # Draw previous measurements with dashed lines
        for qubit_pair in self.prev_measurements:
            q1, q2 = qubit_pair
            meas_type = self.get_measurement_type((q1, q2))
            pos1 = np.argwhere(self.grid == q1)[0]
            pos2 = np.argwhere(self.grid == q2)[0]
            x1, y1 = pos1[1], self.rows - pos1[0]
            x2, y2 = pos2[1], self.rows - pos2[0]
            color = 'blue' if meas_type == 'X' else 'red'
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, linestyle='--', alpha=0.7)
        # Draw selected edges (current measurements)
        for qubit_pair in self.selected_edges:
            q1, q2 = qubit_pair
            meas_type = self.get_measurement_type((q1, q2))
            pos1 = np.argwhere(self.grid == q1)[0]
            pos2 = np.argwhere(self.grid == q2)[0]
            x1, y1 = pos1[1], self.rows - pos1[0]
            x2, y2 = pos2[1], self.rows - pos2[0]
            color = 'blue' if meas_type == 'X' else 'red'
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        self.ax.axis('off')
        self.ax.set_title("Select Edges for Measurements")
        plt.draw()

    def start_interactive_session(self):
        print("Interactive session started. Click on edges to select measurements.")
        print("Close the plot window when you're done selecting.")
        # Update prev_measurements to current_measurements before starting the new session
        self.prev_measurements = self.current_measurements.copy()
        # Reset selected edges at the start
        self.selected_edges = []
        # Create a new figure and axes for the interactive session
        self.fig, self.ax = plt.subplots(figsize=(self.cols + 1, self.rows + 1))
        self.update_interactive_plot()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        self.fig.canvas.mpl_disconnect(self.cid)
        plt.close(self.fig)
        # After the window is closed, run the time step with selected measurements
        self.run_time_step(self.selected_edges.copy())

# Example usage
if __name__ == "__main__":
    size = 7
    grid_size = size - 1
    code = BaconShorCode(grid_size, grid_size)
    while True:
        code.start_interactive_session()
        
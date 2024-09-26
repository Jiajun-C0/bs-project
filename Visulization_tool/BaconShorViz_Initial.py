import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class Stabilizer:
    def __init__(self, qubits_ops):
        self.qubits_ops = qubits_ops  # {qubit_index: 'X' or 'Z'}

    def commutes_with(self, measurement):
        anti_commuting_qubits = 0
        for q in measurement['qubits']:
            op_stab = self.qubits_ops.get(q, 'I')
            op_meas = measurement['type']
            if op_stab != 'I' and op_stab != op_meas:
                anti_commuting_qubits += 1
        return anti_commuting_qubits % 2 == 0

    def multiply(self, other):
        new_qubits_ops = self.qubits_ops.copy()
        for q, op in other.qubits_ops.items():
            if q in new_qubits_ops:
                if new_qubits_ops[q] == op:
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
        self.num_qubits = (rows + 1) * (cols + 1)
        self.grid = np.arange(1, self.num_qubits + 1).reshape((rows + 1, cols + 1))
        self.stabilizer_generators = set()
        self.time_step = 0
        self.prev_measurements = []
        self.current_measurements = []
        self.print_qubit_grid()

    def print_qubit_grid(self):
        print("Qubit Grid Positions:")
        for i in range(self.rows + 1):
            row = ''
            for j in range(self.cols + 1):
                q = self.grid[i, j]
                row += f'{q:2d} '
            print(row)
        print("\nQubit positions are given by (row, column) indices starting from (0,0).")
        print("Use these indices to determine adjacent qubits for measurements.\n")

    def is_valid_measurement(self, qubit_pair):
        q1, q2 = qubit_pair
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
        # Store previous and current measurements
        self.prev_measurements = self.current_measurements
        self.current_measurements = valid_measurements
        self.draw_grid()

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
            self.stabilizer_generators = new_stabilizers
        else:
            self.stabilizer_generators |= new_stabilizers
            self.adjust_stabilizers(measurement_stabilizers)

        print("Stabilizer Generators:")
        for stab in self.stabilizer_generators:
            print(str(stab))

    def adjust_stabilizers(self, measurement_stabilizers):
        for meas_stab in measurement_stabilizers:
            anti_commuting_stabs = [stab for stab in self.stabilizer_generators if not stab.commutes_with({'qubits': list(meas_stab.qubits_ops.keys()), 'type': list(meas_stab.qubits_ops.values())[0]}) and stab != meas_stab]
            while len(anti_commuting_stabs) >= 2:
                stab1 = anti_commuting_stabs.pop()
                stab2 = anti_commuting_stabs.pop()
                combined_stab = stab1.multiply(stab2)
                self.stabilizer_generators.discard(stab1)
                self.stabilizer_generators.discard(stab2)
                if not combined_stab.is_trivial():
                    self.stabilizer_generators.add(combined_stab)
                anti_commuting_stabs = [stab for stab in self.stabilizer_generators if not stab.commutes_with({'qubits': list(meas_stab.qubits_ops.keys()), 'type': list(meas_stab.qubits_ops.values())[0]}) and stab != meas_stab]
            if len(anti_commuting_stabs) == 1:
                stab = anti_commuting_stabs[0]
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
            positions = [np.argwhere(self.grid == q)[0] for q in stab.qubits_ops.keys()]
            xs = [pos[1] for pos in positions]
            ys = [self.rows - pos[0] for pos in positions]
            # Determine color based on stabilizer type
            ops = set(stab.qubits_ops.values())
            if ops == {'X'}:
                color = 'lightblue'
            elif ops == {'Z'}:
                color = 'lightpink'
            else:
                color = 'lightgreen'  # For mixed operators
            if len(xs) == 4:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                rect = Rectangle((min_x, min_y), 1, 1, linewidth=1, edgecolor='none', facecolor=color, alpha=0.5)
                ax.add_patch(rect)
            elif len(xs) == 2:
                # For stabilizers of length 2, draw as per usual
                ax.plot(xs, ys, color=color, linewidth=2)
            else:
                ax.plot(xs, ys, 's', color=color, markersize=8, alpha=0.5)
        plt.axis('off')
        plt.title(f"Time Step {self.time_step}")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize a 4x4 grid (which means 5x5 vertices)
    size = 5
    grid_size = size - 1
    code = BaconShorCode(grid_size, grid_size)

    while True:
        try:
            # Input measurements from the user
            user_input = input("Enter pair measurements for this time step (format: 1,2;3,4;... or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            
            # Process the input into a list of pairs
            measurements = [list(map(int, pair.split(','))) for pair in user_input.split(';')]
            code.run_time_step(measurements)

        except ValueError:
            print("Invalid input. Please enter measurements in the correct format.")
        except Exception as e:
            print(f"An error occurred: {e}")

# # Example usage
# if __name__ == "__main__":
#     # Initialize a 4x4 grid (which means 5x5 vertices)
#     size = 5
#     grid_size = size - 1
#     code = BaconShorCode(grid_size, grid_size)
#     # Time step 1 measurements
#     measurements_t1 = [[1,2],[3,4],[6,7],[8,9]]
#     code.run_time_step(measurements_t1)
#     # Time step 2 measurements
#     measurements_t2 = [[1,6],[2,7],[3,8],[4,9],[11,12],[13,14]]
#     code.run_time_step(measurements_t2)
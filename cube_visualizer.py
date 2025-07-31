#!/usr/bin/env python3
"""
Rubik's Cube Visual Interface - Design Dexterity Challenge
Interactive 3D visualization and solving interface using matplotlib.

Features:
- 3D cube visualization with color mapping
- Interactive move execution
- Solving animation
- State tracking and history
- Performance metrics display
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from rubiks_cube_solver import RubiksCube, CubeSolver, Move, Face, PerformanceAnalyzer
import time

class CubeVisualizer:
    """
    3D Rubik's Cube visualizer with interactive controls and solving animation
    """
    
    def __init__(self):
        self.cube = RubiksCube()
        self.solver = CubeSolver()
        self.fig = None
        self.ax = None
        self.solving_animation = None
        self.solution_moves = []
        self.current_move_index = 0
        
        # Color mapping for visualization
        self.face_colors = {
            0: '#00FF00',  # Green
            1: '#FF0000',  # Red
            2: '#0000FF',  # Blue
            3: '#FFA500',  # Orange
            4: '#FFFFFF',  # White
            5: '#FFFF00'   # Yellow
        }
        
        # Face names for display
        self.face_names = {
            Face.FRONT: 'Front (Green)',
            Face.RIGHT: 'Right (Red)',
            Face.BACK: 'Back (Blue)',
            Face.LEFT: 'Left (Orange)',
            Face.UP: 'Up (White)',
            Face.DOWN: 'Down (Yellow)'
        }
        
    def create_interface(self):
        """Create the main visualization interface"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Rubik\'s Cube Solver - Design Dexterity Challenge', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 3D cube visualization
        self.ax_3d = self.fig.add_subplot(gs[:2, :2], projection='3d')
        self.ax_3d.set_title('3D Cube Visualization', fontweight='bold')
        
        # 2D face displays
        self.ax_faces = {}
        face_positions = [
            (0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)
        ]
        
        for i, face in enumerate(Face):
            row, col = face_positions[i]
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_title(self.face_names[face], fontsize=10)
            self.ax_faces[face] = ax
            
        # Controls and information panel
        self.ax_info = self.fig.add_subplot(gs[2, :2])
        self.ax_info.axis('off')
        
        # Initialize displays
        self.update_display()
        
    def draw_3d_cube(self):
        """Draw the 3D representation of the cube"""
        self.ax_3d.clear()
        self.ax_3d.set_title('3D Cube Visualization', fontweight='bold')
        
        # Define face positions and orientations
        face_positions = {
            Face.FRONT: (0, 0, 1),   # Z = 1
            Face.BACK: (0, 0, -1),   # Z = -1
            Face.RIGHT: (1, 0, 0),   # X = 1
            Face.LEFT: (-1, 0, 0),   # X = -1
            Face.UP: (0, 1, 0),      # Y = 1
            Face.DOWN: (0, -1, 0)    # Y = -1
        }
        
        # Draw each face
        for face, (x_offset, y_offset, z_offset) in face_positions.items():
            self._draw_face_3d(face, x_offset, y_offset, z_offset)
            
        # Set equal aspect ratio and labels
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        
        # Set viewing angle
        self.ax_3d.view_init(elev=20, azim=45)
        
        # Set axis limits
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-2, 2])
        
    def _draw_face_3d(self, face: Face, x_offset, y_offset, z_offset):
        """Draw a single face in 3D space"""
        face_array = self.cube.faces[face]
        
        for i in range(3):
            for j in range(3):
                color_id = face_array[i, j]
                color = self.face_colors[color_id]
                
                # Calculate sticker position
                if face in [Face.FRONT, Face.BACK]:
                    x = j - 1 + x_offset
                    y = 1 - i + y_offset
                    z = z_offset
                    verts = [
                        [x-0.4, y-0.4, z], [x+0.4, y-0.4, z],
                        [x+0.4, y+0.4, z], [x-0.4, y+0.4, z]
                    ]
                elif face in [Face.LEFT, Face.RIGHT]:
                    x = x_offset
                    y = 1 - i + y_offset
                    z = j - 1 + z_offset
                    verts = [
                        [x, y-0.4, z-0.4], [x, y-0.4, z+0.4],
                        [x, y+0.4, z+0.4], [x, y+0.4, z-0.4]
                    ]
                else:  # UP, DOWN
                    x = j - 1 + x_offset
                    y = y_offset
                    z = 1 - i + z_offset
                    verts = [
                        [x-0.4, y, z-0.4], [x+0.4, y, z-0.4],
                        [x+0.4, y, z+0.4], [x-0.4, y, z+0.4]
                    ]
                
                # Create and add the face patch
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly = Poly3DCollection([verts], alpha=0.9)
                poly.set_facecolor(color)
                poly.set_edgecolor('black')
                poly.set_linewidth(2)
                self.ax_3d.add_collection3d(poly)
                
    def draw_2d_faces(self):
        """Draw 2D representation of all cube faces"""
        for face in Face:
            ax = self.ax_faces[face]
            ax.clear()
            ax.set_title(self.face_names[face], fontsize=10)
            
            face_array = self.cube.faces[face]
            
            # Create grid of colored squares
            for i in range(3):
                for j in range(3):
                    color_id = face_array[i, j]
                    color = self.face_colors[color_id]
                    
                    # Create square patch
                    square = patches.Rectangle((j, 2-i), 1, 1, 
                                             facecolor=color, 
                                             edgecolor='black', 
                                             linewidth=2)
                    ax.add_patch(square)
            
            # Set axis properties
            ax.set_xlim(0, 3)
            ax.set_ylim(0, 3)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
    def update_info_panel(self):
        """Update the information panel with current state and controls"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Current state information
        info_text = f"CUBE STATUS:\n"
        info_text += f"Solved: {'✓ YES' if self.cube.is_solved() else '✗ NO'}\n"
        info_text += f"Move History Length: {len(self.cube.move_history)}\n"
        info_text += f"State Signature: {self.cube.get_state_signature()[:20]}...\n\n"
        
        # Last few moves
        recent_moves = self.cube.move_history[-10:] if self.cube.move_history else []
        info_text += f"Recent Moves: {[move.value for move in recent_moves]}\n\n"
        
        # Controls information
        info_text += "CONTROLS:\n"
        info_text += "• Press 'r' to scramble cube\n"
        info_text += "• Press 's' to solve cube\n"
        info_text += "• Press 'a' to animate solution\n"
        info_text += "• Press 'c' to clear/reset cube\n"
        info_text += "• Press 'p' to run performance test\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
        
    def update_display(self):
        """Update all visual components"""
        self.draw_3d_cube()
        self.draw_2d_faces()
        self.update_info_panel()
        
        if self.fig:
            self.fig.canvas.draw()
            
    def scramble_cube(self, num_moves=25):
        """Scramble the cube and update display"""
        print(f"Scrambling cube with {num_moves} moves...")
        scramble_moves = self.cube.scramble(num_moves)
        print(f"Scramble: {[move.value for move in scramble_moves]}")
        self.update_display()
        
    def solve_cube(self):
        """Solve the cube and update display"""
        if self.cube.is_solved():
            print("Cube is already solved!")
            return
            
        print("Solving cube...")
        start_time = time.time()
        
        # Create a copy for solving
        solve_cube = self.cube.copy()
        solution = self.solver.solve(solve_cube)
        
        solve_time = time.time() - start_time
        
        print(f"Solution found in {solve_time:.3f} seconds!")
        print(f"Solution: {[move.value for move in solution]}")
        print(f"Number of moves: {len(solution)}")
        
        # Apply solution to display cube
        self.cube.execute_sequence(solution)
        self.solution_moves = solution
        self.update_display()
        
        # Display statistics
        stats = self.solver.get_statistics()
        print(f"Algorithm steps: {', '.join(stats['algorithm_steps'])}")
        
    def animate_solution(self):
        """Animate the solving process"""
        if not self.solution_moves:
            print("No solution to animate. Solve the cube first.")
            return
            
        # Reset cube to pre-solved state
        # This is a simplified version - in practice, we'd store the scrambled state
        self.cube = RubiksCube()
        self.scramble_cube(25)  # Re-scramble for animation
        
        self.current_move_index = 0
        
        def animate_step(frame):
            if self.current_move_index < len(self.solution_moves):
                move = self.solution_moves[self.current_move_index]
                self.cube.execute_move(move)
                self.update_display()
                
                # Update title with current move
                self.ax_3d.set_title(f'Solving: Move {self.current_move_index + 1}/{len(self.solution_moves)} - {move.value}',
                                   fontweight='bold')
                
                self.current_move_index += 1
            else:
                self.solving_animation.event_source.stop()
                self.ax_3d.set_title('Solution Complete!', fontweight='bold')
                
        # Create animation
        self.solving_animation = FuncAnimation(self.fig, animate_step, 
                                             interval=500, repeat=False)
        
    def reset_cube(self):
        """Reset cube to solved state"""
        self.cube = RubiksCube()
        self.solution_moves = []
        self.current_move_index = 0
        print("Cube reset to solved state.")
        self.update_display()
        
    def run_performance_test(self):
        """Run performance benchmark and display results"""
        print("Running performance benchmark (10 tests)...")
        analyzer = PerformanceAnalyzer()
        
        # Run benchmark
        results = analyzer.benchmark_solver(10)
        
        # Display results
        print("\nBENCHMARK RESULTS:")
        print(f"Success Rate: {results['successful_solves']}/{results['total_tests']}")
        
        if results['successful_solves'] > 0:
            print(f"Average Moves: {results['average_moves']:.1f}")
            print(f"Average Time: {results['average_time']:.3f} seconds")
            print(f"Move Range: {results['min_moves']} - {results['max_moves']}")
            
        # Generate full report
        report = analyzer.generate_report()
        print("\n" + report)
        
    def on_key_press(self, event):
        """Handle keyboard input for interactive controls"""
        if event.key == 'r':
            self.scramble_cube()
        elif event.key == 's':
            self.solve_cube()
        elif event.key == 'a':
            self.animate_solution()
        elif event.key == 'c':
            self.reset_cube()
        elif event.key == 'p':
            self.run_performance_test()
        elif event.key == 'q':
            plt.close(self.fig)
            
    def show(self):
        """Display the interactive interface"""
        self.create_interface()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Display instructions
        print("\nRUBIK'S CUBE INTERACTIVE SOLVER")
        print("=" * 40)
        print("Controls:")
        print("  R - Scramble cube")
        print("  S - Solve cube")
        print("  A - Animate solution")
        print("  C - Clear/Reset cube")
        print("  P - Performance test")
        print("  Q - Quit")
        print("\nClick on the figure and use keyboard controls!")
        
        plt.show()


class CubeAnalyzer:
    """Advanced analysis tools for cube solving algorithms"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_scramble_complexity(self, num_tests=50):
        """Analyze how scramble complexity affects solving difficulty"""
        print("Analyzing scramble complexity...")
        
        scramble_lengths = [10, 15, 20, 25, 30, 35, 40]
        results = {}
        
        for length in scramble_lengths:
            print(f"Testing scramble length: {length}")
            
            total_moves = 0
            total_time = 0
            successful_solves = 0
            
            for _ in range(num_tests):
                cube = RubiksCube()
                cube.scramble(length)
                
                solver = CubeSolver()
                start_time = time.time()
                solution = solver.solve(cube)
                solve_time = time.time() - start_time
                
                # Verify solution
                test_cube = RubiksCube()
                test_cube.execute_sequence(cube.move_history[-length:])  # Get scramble
                test_cube.execute_sequence(solution)
                
                if test_cube.is_solved():
                    successful_solves += 1
                    total_moves += len(solution)
                    total_time += solve_time
                    
            if successful_solves > 0:
                results[length] = {
                    'avg_moves': total_moves / successful_solves,
                    'avg_time': total_time / successful_solves,
                    'success_rate': successful_solves / num_tests
                }
            else:
                results[length] = {
                    'avg_moves': 0,
                    'avg_time': 0,
                    'success_rate': 0
                }
                
        self.analysis_results['scramble_complexity'] = results
        return results
        
    def compare_algorithms(self):
        """Compare different solving approaches"""
        # This would implement different solving algorithms
        # For now, we'll analyze the current algorithm's characteristics
        
        print("Analyzing algorithm characteristics...")
        
        # Test various cube states
        test_cases = [
            "solved",
            "single_move",
            "corner_twist",
            "edge_flip",
            "complex_scramble"
        ]
        
        results = {}
        
        for case in test_cases:
            cube = RubiksCube()
            
            if case == "solved":
                pass  # Already solved
            elif case == "single_move":
                cube.execute_move(Move.R)
            elif case == "corner_twist":
                cube.execute_sequence([Move.R, Move.U, Move.R_PRIME, Move.U_PRIME])
            elif case == "edge_flip":
                cube.execute_sequence([Move.F, Move.R, Move.U_PRIME, Move.R_PRIME, Move.F_PRIME])
            elif case == "complex_scramble":
                cube.scramble(30)
                
            solver = CubeSolver()
            start_time = time.time()
            solution = solver.solve(cube)
            solve_time = time.time() - start_time
            
            results[case] = {
                'moves': len(solution),
                'time': solve_time,
                'solution': [move.value for move in solution]
            }
            
        self.analysis_results['algorithm_comparison'] = results
        return results
        
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        report = "ADVANCED CUBE SOLVING ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        if 'scramble_complexity' in self.analysis_results:
            report += "SCRAMBLE COMPLEXITY ANALYSIS:\n"
            report += "-" * 30 + "\n"
            
            for length, data in self.analysis_results['scramble_complexity'].items():
                report += f"Scramble Length {length:2d}: "
                report += f"Avg Moves: {data['avg_moves']:5.1f}, "
                report += f"Avg Time: {data['avg_time']:6.3f}s, "
                report += f"Success: {data['success_rate']*100:5.1f}%\n"
            report += "\n"
            
        if 'algorithm_comparison' in self.analysis_results:
            report += "ALGORITHM PERFORMANCE BY CASE TYPE:\n"
            report += "-" * 35 + "\n"
            
            for case, data in self.analysis_results['algorithm_comparison'].items():
                report += f"{case:15s}: {data['moves']:3d} moves, {data['time']:6.3f}s\n"
            report += "\n"
            
        report += "OPTIMIZATION RECOMMENDATIONS:\n"
        report += "-" * 30 + "\n"
        report += "1. Implement more efficient OLL/PLL recognition\n"
        report += "2. Add move optimization to reduce redundant sequences\n"
        report += "3. Implement lookahead for F2L optimization\n"
        report += "4. Add state prediction for better path planning\n"
        report += "5. Consider implementing Kociemba's algorithm for optimality\n"
        
        return report


def main():
    """Main function to run the cube visualizer"""
    print("Starting Rubik's Cube Visual Interface...")
    
    # Create and show visualizer
    visualizer = CubeVisualizer()
    
    # Optional: Run some initial tests
    print("\nRunning initial demonstration...")
    
    # Create a scrambled cube for demo
    visualizer.scramble_cube(15)
    
    # Show the interface
    visualizer.show()


if __name__ == "__main__":
    main()
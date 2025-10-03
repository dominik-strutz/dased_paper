import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.widgets import Button
import pickle
import os
from shapely.plotting import plot_polygon

class PointSelector:
    def __init__(self, design_space, obstacles=None, cable_properties=None, figsize=(12, 8)):
        """
        Initialize the PointSelector with all necessary inputs.
        
        Parameters:
        -----------
        design_space : shapely.geometry object
            The design space boundary (should have .exterior.xy)
        obstacles : shapely.geometry.collection object, optional
            Collection of obstacle geometries (should have .geoms)
        cable_properties : dict, optional
            Dictionary containing at least 'fixed_points' key with list of [x, y] coordinates
        figsize : tuple, optional
            Figure size as (width, height)
        """
        # Store the original backend
        self.original_backend = mpl.get_backend()
        mpl.use('QtAgg')
        
        self.design_space = design_space
        self.obstacles = obstacles
        self.cable_properties = cable_properties or {}
        self.figsize = figsize
        
        self.selected_points = []
        self.current_line = []
        self.current_line_index = 0
        self.colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        self.result = None  # Store the final result
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.info_text = None
        
        # Connect close event to restore backend and return points
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.plot_setup()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.setup_buttons()
        self.update_info_display()
        plt.show()

    def on_close(self, event):
        """Restore the original backend and prepare result when the plot is closed"""
        # Finish current line if it exists
        if self.current_line:
            self.selected_points.append(self.current_line.copy())
        
        # Store the result
        self.result = self.selected_points.copy()
        
        # Restore backend
        mpl.use(self.original_backend)

    def get_result(self):
        """Return the selected points after the window is closed"""
        return self.result

    def get_current_color(self):
        return self.colors[self.current_line_index % len(self.colors)]

    def calculate_line_length(self, line_points):
        """Calculate total length of a line including connection to fixed point"""
        if not line_points:
            return 0.0
        
        total_length = 0.0
        
        # Only calculate connection to fixed point if cable_properties is provided
        if self.cable_properties and 'fixed_points' in self.cable_properties:
            fixed_point = self.cable_properties['fixed_points'][0]
            
            # Distance from fixed point to first point
            if len(line_points) > 0:
                first_point = line_points[0]
                total_length += np.sqrt((first_point[0] - fixed_point[0])**2 + (first_point[1] - fixed_point[1])**2)
        
        # Distances between consecutive points
        for i in range(1, len(line_points)):
            p1, p2 = line_points[i-1], line_points[i]
            total_length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        return total_length

    def update_info_display(self):
        """Update the information display showing current line stats"""
        if self.info_text:
            self.info_text.remove()
        
        num_points = len(self.current_line)
        line_length = self.calculate_line_length(self.current_line)
        
        info_str = f"Current Line:\nPoints: {num_points}\nLength: {line_length:.1f} m"
        
        # Position text on the right side of the plot
        self.info_text = self.fig.text(0.88, 0.7, info_str, fontsize=10, 
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                                      verticalalignment='top')

    def plot_setup(self):
        # Plot design space
        plot_polygon(self.design_space, ax=self.ax,
                     facecolor='none', add_points=False,
                     edgecolor='k', alpha=0.5, label='Design Space')
        
        # Plot obstacles if provided
        if self.obstacles:
            for i, obstacle in enumerate(self.obstacles.geoms):
                ox, oy = obstacle.exterior.xy
                self.ax.plot(ox, oy, 'r-', linewidth=1.5, alpha=0.7)
                self.ax.fill(ox, oy, color='red', alpha=0.3, label='Obstacles' if i == 0 else '')
        
        # Plot fixed points if provided
        if self.cable_properties and 'fixed_points' in self.cable_properties:
            fixed_point = self.cable_properties['fixed_points'][0]
            self.ax.plot(fixed_point[0], fixed_point[1], 'go', markersize=8, label='Fixed Point')
        
        self.ax.set_xlabel('Easting (m)')
        self.ax.set_ylabel('Northing (m)')
        self.ax.set_title('DAS Layout Design - Click to select points')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Move legend below plot
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    def setup_buttons(self):
        ax_new_line = plt.axes([0.02, 0.85, 0.1, 0.04])
        ax_finish_line = plt.axes([0.02, 0.80, 0.1, 0.04])
        ax_clear_current = plt.axes([0.02, 0.75, 0.12, 0.04]) # New button
        ax_clear_all = plt.axes([0.02, 0.70, 0.1, 0.04]) # Shifted down
        ax_print = plt.axes([0.02, 0.65, 0.1, 0.04]) # Shifted down
        
        self.btn_new_line = Button(ax_new_line, 'New Line')
        self.btn_finish_line = Button(ax_finish_line, 'Finish Line')
        self.btn_clear_current = Button(ax_clear_current, 'Clear Current Line') # New button
        self.btn_clear_all = Button(ax_clear_all, 'Clear All')
        self.btn_print = Button(ax_print, 'Print Points')
        
        self.btn_new_line.on_clicked(self.new_line)
        self.btn_finish_line.on_clicked(self.finish_line)
        self.btn_clear_current.on_clicked(self.clear_current_line) # New connection
        self.btn_clear_all.on_clicked(self.clear_all)
        self.btn_print.on_clicked(self.print_points)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        current_color = self.get_current_color()
        self.current_line.append([event.xdata.item(), event.ydata.item()])
        self.ax.plot(event.xdata, event.ydata, 'o', color=current_color, markersize=5)
        
        # Connect to fixed point if this is the first point of a line and fixed points exist
        if (len(self.current_line) == 1 and 
            self.cable_properties and 
            'fixed_points' in self.cable_properties):
            fixed_point = self.cable_properties['fixed_points'][0]
            self.ax.plot([fixed_point[0], event.xdata], [fixed_point[1], event.ydata], 
                        '--', color=current_color, linewidth=1.5, alpha=0.7)
        
        # Connect consecutive points in the same line
        if len(self.current_line) > 1:
            x_coords = [p[0] for p in self.current_line[-2:]]
            y_coords = [p[1] for p in self.current_line[-2:]]
            self.ax.plot(x_coords, y_coords, '-', color=current_color, linewidth=2)
        
        self.update_info_display()
        self.fig.canvas.draw()

    def new_line(self, event):
        if self.current_line:
            self.finish_line(event)

    def finish_line(self, event):
        if self.current_line:
            self.selected_points.append(self.current_line.copy())
            self.current_line = []
            self.current_line_index += 1
            self.update_info_display()
            self.fig.canvas.draw()
            print(f"Line finished. Total lines: {len(self.selected_points)}")

    def clear_current_line(self, event):
        if not self.current_line:
            print("No current line to clear.")
            return

        self.current_line = []
        
        self.ax.clear()
        self.plot_setup() # Redraws background, grid, labels, legend etc.
        
        # Re-plot all finished lines
        for line_idx, finished_line_points in enumerate(self.selected_points):
            if not finished_line_points:
                continue
            
            line_color = self.colors[line_idx % len(self.colors)]
            
            # Plot connection from fixed point to first point of this finished line
            if self.cable_properties and 'fixed_points' in self.cable_properties:
                fixed_point = self.cable_properties['fixed_points'][0]
                if finished_line_points: # Ensure line is not empty
                    first_point = finished_line_points[0]
                    self.ax.plot([fixed_point[0], first_point[0]], [fixed_point[1], first_point[1]], 
                                 '--', color=line_color, linewidth=1.5, alpha=0.7)

            # Plot points of this finished line
            for point_idx, point in enumerate(finished_line_points):
                self.ax.plot(point[0], point[1], 'o', color=line_color, markersize=5)
                # Plot connections between consecutive points in this finished line
                if point_idx > 0:
                    p1 = finished_line_points[point_idx-1]
                    p2 = point
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=line_color, linewidth=2)
        
        self.update_info_display()
        self.fig.canvas.draw()
        print("Current line cleared.")

    def clear_all(self, event):
        self.selected_points = []
        self.current_line = []
        self.current_line_index = 0
        self.ax.clear()
        self.plot_setup()
        self.update_info_display()
        self.fig.canvas.draw()
        print("All points cleared")

    def print_points(self, event):
        if self.current_line: # Finish the current line before printing, if any points exist
            self.finish_line(event) # This will also update selected_points
        
        print("\nSelected points (list of lists):")
        # Ensure we are printing the latest state of selected_points
        points_to_print = self.selected_points
        if not points_to_print and self.result: # If window was closed, result might be more up-to-date
             points_to_print = self.result

        print(points_to_print)
        print(f"\nTotal number of lines: {len(points_to_print)}")
        for i, line in enumerate(points_to_print):
            length = self.calculate_line_length(line)
            print(f"Line {i+1}: {len(line)} points, Length: {length:.1f} m")


def select_points(design_space, obstacles=None, cable_properties=None, figsize=(12, 8), filename=None):
    """
    Convenience function to create a PointSelector and return the selected points.
    
    Parameters:
    -----------
    design_space : shapely.geometry object
        The design space boundary
    obstacles : shapely.geometry.collection object, optional
        Collection of obstacle geometries
    cable_properties : dict, optional
        Dictionary containing cable properties
    figsize : tuple, optional
        Figure size as (width, height)
    filename : str, optional
        Path to file for loading/saving proposal points. If None, no file operations are performed.
    
    Returns:
    --------
    list of lists
        Each inner list contains [x, y] coordinates for points in a line
    """
    
    # Try to load precomputed points if filename is provided
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                points = pickle.load(f)
            print(f"Loaded precomputed points from {filename}")
            return points
        except Exception as e:
            print(f"Failed to load points from {filename}: {e}")
            print("Proceeding to generate new points...")
    
    # Generate new points using the selector
    selector = PointSelector(design_space, obstacles, cable_properties, figsize)
    points = selector.get_result() # This will block until the window is closed
    
    # Save points if filename is provided and points were generated
    if filename and points: # points might be None if window closed immediately
        try:
            # Create directory if it doesn't exist
            # Ensure filename is a valid path before trying to get dirname
            if os.path.dirname(filename): # Check if dirname is not empty
                 os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(points, f)
            print(f"Saved points to {filename}")
        except Exception as e:
            print(f"Failed to save points to {filename}: {e}")
    
    return points
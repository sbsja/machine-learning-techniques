from dtree import *
import random
import monkdata as m
from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np
import sys

class TestErrorPlot(QtWidgets.QMainWindow):
    def __init__(self, fractions):
        super().__init__()

        self.setWindowTitle(f"Test Error vs. Training Fraction ({fractions['Monk_Name']})")
        self.setGeometry(100, 100, 800, 600)

        # Create the plot widget
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")  # White background
        self.plot_graph.setTitle("Test Error vs. Training Fraction", color="black", size="14pt")

        # Axis labels
        styles = {"color": "black", "font-size": "12px"}
        self.plot_graph.setLabel("left", "Test Error", **styles)
        self.plot_graph.setLabel("bottom", "Training Fraction", **styles)

        # Enable grid
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.addLegend()

        # Extract data
        x_values = sorted(fractions.keys() - {"Monk_Name"})  # Remove non-numeric key
        y_means = []
        y_upper = []
        y_lower = []

        for frac in x_values:
            test_errors = [1 - acc for acc in fractions[frac]["Test_accuracies"]]  # Convert accuracy to error
            mean_error = np.mean(test_errors)
            std_error = np.std(test_errors)
            print(mean_error,std_error,frac)
            y_means.append(mean_error)
            y_upper.append(mean_error + std_error)  # Upper bound (+1 Std)
            y_lower.append(mean_error - std_error)  # Lower bound (-1 Std)

        # Plot Mean Test Error (Blue Line)
        pen_mean = pg.mkPen(color=(0, 0, 255), width=2)  # Blue line
        self.plot_graph.plot(x_values, y_means, name="Mean Test Error", pen=pen_mean, symbol="o", symbolSize=8, symbolBrush="b")

        # Plot Upper Bound (+1 Std) (Green Line)
        pen_upper = pg.mkPen(color=(0, 128, 0), width=2, style=pg.QtCore.Qt.DashLine)  # Green dashed line
        self.plot_graph.plot(x_values, y_upper, name="Upper Bound (+1 Std)", pen=pen_upper, symbol="t", symbolSize=8, symbolBrush="g")

        # Plot Lower Bound (-1 Std) (Red Line)
        pen_lower = pg.mkPen(color=(255, 0, 0), width=2, style=pg.QtCore.Qt.DashLine)  # Red dashed line
        self.plot_graph.plot(x_values, y_lower, name="Lower Bound (-1 Std)", pen=pen_lower, symbol="t", symbolSize=8, symbolBrush="r")


def plot_test_error(fractions):
    """Launch PyQt5 window with PyQtGraph for better visualization."""
    app = QtWidgets.QApplication(sys.argv)
    window = TestErrorPlot(fractions)
    window.show()
    sys.exit(app.exec_())



def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return (ldata[:breakPoint], ldata[breakPoint:])



def pruning(train_set, val_set):
    decision_tree = buildTree(train_set, m.attributes,maxdepth=6)
    best_tree = decision_tree
    best_accuracy = check(best_tree, val_set)  # Original tree accuracy
    improved=True

    while improved:
        improved = False  # Track if pruning improves accuracy
        # Generate all pruned trees
        for pruned_tree in allPruned(best_tree):
            pruned_accuracy = check(pruned_tree, val_set)

            # Keep the pruned tree if it improves accuracy
            if pruned_accuracy >= best_accuracy:
                best_tree = pruned_tree
                best_accuracy = pruned_accuracy
                improved = True  # Mark that we improved

    return best_tree,best_accuracy

def eval_fracs(fractions):
    for frac, frac_dict in fractions.items():
        if frac == "Monk_Name":
            continue
        for i in range(500):
            print(i)
            train_set, val_set = partition(m.monk1, frac)
            # Now train_set and val_set can be used properly
            best_tree, best_accuracy= pruning(train_set, val_set)
            frac_dict["Best_Trees"].append(best_tree)
            frac_dict["Val_accuracies"].append(best_accuracy)
            frac_dict["Test_accuracies"].append(check(best_tree, m.monk1test))

def create_fracs():
    M1_fractions = {
        0.3 + i / 10: {"Test_set": m.monk1test, "Best_Trees": [], "Val_accuracies": [], "Test_accuracies": []}
        for i in range(6)}
    M1_fractions["Monk_Name"] = "MONK-1"


    M3_fractions = {
        0.3 + i / 10: {"Test_set": m.monk3test, "Best_Trees": [],"Val_accuracies": [], "Test_accuracies": []}
        for i in range(6)}

    M3_fractions["Monk_Name"] = "MONK-3"

    return M1_fractions, M3_fractions


def main():
    M1_fractions, M3_fractions = create_fracs()
    eval_fracs(M1_fractions)
    #eval_fracs(M3_fractions)
    plot_test_error(M1_fractions)
    #plot_test_error(M3_fractions)


if __name__ == "__main__":
    main()












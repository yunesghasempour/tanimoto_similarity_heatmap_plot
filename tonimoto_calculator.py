#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QTextEdit, QPushButton, QTabWidget, QSplitter, 
                            QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog, QSizePolicy, QMenu,
                            QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, rdDepictor
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA
import os

# Define main colors
YELLOW = "#ffd700"
BLACK = "#1a1a1a"
DARK_YELLOW = "#ccac00"
LIGHT_BLACK = "#333333"
WHITE = "#ffffff"
GRAY = "#cccccc"

class MoleculePreviewWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BLACK};
                color: {YELLOW};
                border: 1px solid {YELLOW};
                border-radius: 4px;
            }}
            QLabel {{
                color: {YELLOW};
                font-weight: bold;
            }}
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add title
        title = QLabel("Molecule Structures")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create molecule displays
        self.mol1_label = QLabel()
        self.mol2_label = QLabel()
        self.mol1_label.setAlignment(Qt.AlignCenter)
        self.mol2_label.setAlignment(Qt.AlignCenter)
        
        # Add molecule labels
        layout.addWidget(self.mol1_label)
        layout.addWidget(self.mol2_label)
        
        # Set fixed size
        self.setFixedSize(600, 600)
    
    def update_molecules(self, mol1, mol2):
        """Update molecule displays"""
        def prepare_mol(mol):
            # Make a copy of the molecule
            mol = Chem.Mol(mol)
            
            # Compute 2D coordinates if they don't exist
            if mol.GetNumConformers() == 0:
                rdDepictor.Compute2DCoords(mol)
            
            # Optimize the 2D depiction
            rdDepictor.GenerateDepictionMatching2DStructure(mol, mol)
            
            return mol
        
        # Prepare molecules
        mol1 = prepare_mol(mol1)
        mol2 = prepare_mol(mol2)
        
        try:
            # Try using Cairo first (better quality)
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.drawOptions().additionalAtomLabelPadding = 0.3
            drawer.drawOptions().bondLineWidth = 2
            drawer.drawOptions().padding = 0.2
            drawer.drawOptions().explicitMethyl = True
            drawer.drawOptions().includeMetadata = False
            
            # Draw first molecule
            drawer.DrawMolecule(mol1)
            drawer.FinishDrawing()
            img1_data = drawer.GetDrawingText()
            
            # Clear and draw second molecule
            drawer.ClearDrawing()
            drawer.DrawMolecule(mol2)
            drawer.FinishDrawing()
            img2_data = drawer.GetDrawingText()
            
            # Convert to QPixmap and display
            def cairo_to_pixmap(img_data):
                # Create QImage from the raw data
                img = QImage.fromData(img_data)
                return QPixmap.fromImage(img)
            
            self.mol1_label.setPixmap(cairo_to_pixmap(img1_data))
            self.mol2_label.setPixmap(cairo_to_pixmap(img2_data))
            
        except Exception as e:
            # Fallback to RDKit's default drawing method
            img1 = Draw.MolToImage(mol1, size=(400, 400), kekulize=True, fitImage=True, padding=0.2)
            img2 = Draw.MolToImage(mol2, size=(400, 400), kekulize=True, fitImage=True, padding=0.2)
            
            def pil_to_pixmap(pil_image):
                """Convert PIL image to QPixmap"""
                img_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
                qim = QImage(img_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
                return QPixmap.fromImage(qim)
            
            self.mol1_label.setPixmap(pil_to_pixmap(img1))
            self.mol2_label.setPixmap(pil_to_pixmap(img2))

class HeatmapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(BLACK)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(BLACK)
        
        super(HeatmapCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # Connect mouse events
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.molecules = None
        self.similarity_matrix = None
        self.smiles_list = None
        self.preview_widget = None
        self.current_cell = None
        
        # Add instruction label
        self.instruction_label = QLabel("Hover over cells to view molecule structures", self)
        self.instruction_label.setStyleSheet(f"color: {YELLOW}; font-weight: bold;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setGeometry(0, 0, width * dpi, 30)
    
    def on_mouse_move(self, event):
        if event.inaxes is None or self.similarity_matrix is None:
            return
        
        # Get coordinates
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or y < 0 or x >= len(self.similarity_matrix) or y >= len(self.similarity_matrix):
            return
        
        # Update preview widget
        if self.preview_widget:
            self.preview_widget.update_molecules(self.molecules[x], self.molecules[y])
    
    def plot_heatmap(self, similarity_matrix, molecules, smiles_list):
        # Clear previous plot but keep the colorbar
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        
        # Store data for click event
        self.molecules = molecules
        self.similarity_matrix = similarity_matrix
        self.smiles_list = smiles_list
        
        # Set heatmap colors with single colorbar
        cmap = sns.color_palette("YlOrBr", as_cmap=True)
        
        # Plot heatmap with single colorbar
        sns.heatmap(similarity_matrix, annot=True, cmap=cmap, 
                   xticklabels=[f"Mol {i+1}" for i in range(len(molecules))],
                   yticklabels=[f"Mol {i+1}" for i in range(len(molecules))],
                   linewidths=0.5, ax=self.axes, cbar_kws={'label': 'Similarity'})
        
        # Set text and axis colors
        self.axes.set_title("Tanimoto Similarity Matrix", color=YELLOW, fontsize=14)
        self.axes.tick_params(colors=YELLOW)
        
        # Set axis label colors
        for text in self.axes.texts:
            text.set_color(BLACK)
        
        plt.tight_layout()
        self.draw()
    
    def save_heatmap(self, filename):
        """Save heatmap to file"""
        self.fig.savefig(filename, facecolor=BLACK, edgecolor='none', bbox_inches='tight')
    
    def clear(self):
        """Clear the heatmap"""
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        self.molecules = None
        self.similarity_matrix = None
        self.smiles_list = None
        self.draw()

class MoleculeCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(BLACK)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(BLACK)
        
        super(MoleculeCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def clear(self):
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        self.draw()

class ScatterCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(BLACK)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(BLACK)
        
        super(ScatterCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def plot_scatter(self, df, x_col, y_col):
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        
        # رسم نمودار پراکندگی
        self.axes.scatter(df[x_col], df[y_col], c=YELLOW, edgecolor=DARK_YELLOW, s=80, alpha=0.7)
        
        # اضافه کردن برچسب‌ها
        for i, txt in enumerate(df.index):
            self.axes.annotate(txt, (df[x_col].iloc[i], df[y_col].iloc[i]), 
                              color=WHITE, fontsize=8)
        
        # تنظیم عنوان و برچسب‌های محورها
        self.axes.set_title(f"نمودار پراکندگی {x_col} در مقابل {y_col}", color=YELLOW)
        self.axes.set_xlabel(x_col, color=YELLOW)
        self.axes.set_ylabel(y_col, color=YELLOW)
        
        # تنظیم رنگ خطوط شبکه و تیک‌ها
        self.axes.grid(True, linestyle='--', alpha=0.3, color=GRAY)
        self.axes.tick_params(colors=YELLOW)
        
        # تنظیم رنگ خطوط محورها
        for spine in self.axes.spines.values():
            spine.set_color(YELLOW)
        
        self.fig.tight_layout()
        self.draw()

class PCACanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(BLACK)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(BLACK)
        
        super(PCACanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def plot_pca(self, df):
        self.axes.clear()
        self.axes.set_facecolor(BLACK)
        
        # انتخاب ویژگی‌های عددی
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
        
        # اجرای PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[numeric_cols])
        
        # رسم نمودار PCA
        self.axes.scatter(pca_result[:, 0], pca_result[:, 1], c=YELLOW, edgecolor=DARK_YELLOW, s=80, alpha=0.7)
        
        # اضافه کردن برچسب‌ها
        for i, txt in enumerate(df.index):
            self.axes.annotate(txt, (pca_result[i, 0], pca_result[i, 1]), 
                              color=WHITE, fontsize=8)
        
        # تنظیم عنوان و برچسب‌های محورها
        variance_explained = pca.explained_variance_ratio_ * 100
        self.axes.set_title("تحلیل مؤلفه‌های اصلی (PCA)", color=YELLOW)
        self.axes.set_xlabel(f"PC1 ({variance_explained[0]:.1f}%)", color=YELLOW)
        self.axes.set_ylabel(f"PC2 ({variance_explained[1]:.1f}%)", color=YELLOW)
        
        # تنظیم رنگ خطوط شبکه و تیک‌ها
        self.axes.grid(True, linestyle='--', alpha=0.3, color=GRAY)
        self.axes.tick_params(colors=YELLOW)
        
        # تنظیم رنگ خطوط محورها
        for spine in self.axes.spines.values():
            spine.set_color(YELLOW)
        
        self.fig.tight_layout()
        self.draw()

class TonimotoCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.molecules = []
        self.smiles_list = []
        self.similarity_matrix = None
        
        self.init_ui()
    
    def init_ui(self):
        # Set window title and size
        self.setWindowTitle("Molecular Tanimoto Coefficient Calculator")
        self.setGeometry(100, 100, 1600, 900)
        
        # Set style and colors
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {BLACK};
                color: {YELLOW};
            }}
            QLabel {{
                color: {YELLOW};
                font-weight: bold;
            }}
            QPushButton {{
                background-color: {YELLOW};
                color: {BLACK};
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {DARK_YELLOW};
            }}
            QTextEdit, QTableWidget {{
                background-color: {LIGHT_BLACK};
                color: {WHITE};
                border: 1px solid {YELLOW};
                border-radius: 4px;
                padding: 4px;
            }}
            QTabWidget::pane {{
                border: 1px solid {YELLOW};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {LIGHT_BLACK};
                color: {YELLOW};
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {YELLOW};
                color: {BLACK};
                font-weight: bold;
            }}
            QScrollArea {{
                border: none;
            }}
        """)
        
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side layout
        left_layout = QVBoxLayout()
        
        # Input section
        input_layout = QHBoxLayout()
        
        # SMILES input section
        smiles_layout = QVBoxLayout()
        smiles_label = QLabel("Enter SMILES strings (one per line):")
        self.smiles_input = QTextEdit()
        self.smiles_input.setPlaceholderText("Example:\nCC(=O)OC1=CC=CC=C1C(=O)O\nCCO\nCCCC")
        self.smiles_input.setMinimumHeight(100)
        
        # Operation buttons
        buttons_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate Tanimoto Coefficient")
        self.calculate_button.clicked.connect(self.calculate_tonimoto)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_data)
        self.example_button = QPushButton("Load Example")
        self.example_button.clicked.connect(self.load_example)
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        
        buttons_layout.addWidget(self.calculate_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.example_button)
        buttons_layout.addWidget(self.save_button)
        
        smiles_layout.addWidget(smiles_label)
        smiles_layout.addWidget(self.smiles_input)
        smiles_layout.addLayout(buttons_layout)
        
        input_layout.addLayout(smiles_layout)
        
        # Add input section to left layout
        left_layout.addLayout(input_layout)
        
        # Results tabs
        self.tabs = QTabWidget()
        
        # Similarity matrix tab
        self.similarity_tab = QWidget()
        similarity_layout = QVBoxLayout(self.similarity_tab)
        similarity_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create container widget for heatmap
        heatmap_container = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        
        # Heatmap display
        self.heatmap_canvas = HeatmapCanvas(heatmap_container)
        heatmap_layout.addWidget(self.heatmap_canvas)
        
        # Add heatmap container to scroll area
        scroll.setWidget(heatmap_container)
        similarity_layout.addWidget(scroll)
        
        # Add tabs
        self.tabs.addTab(self.similarity_tab, "Similarity Matrix")
        
        # Add tabs to left layout
        left_layout.addWidget(self.tabs)
        
        # Add left layout to main layout
        main_layout.addLayout(left_layout)
        
        # Create preview widget
        self.preview_widget = MoleculePreviewWidget(self)
        main_layout.addWidget(self.preview_widget)
        
        # Connect preview widget to heatmap canvas
        self.heatmap_canvas.preview_widget = self.preview_widget
        
        # Show window
        self.show()
    
    def load_example(self):
        """Load example SMILES strings"""
        example_smiles = """CC(=O)OC1=CC=CC=C1C(=O)O    # Aspirin
CCO    # Ethanol
C1=CC=CC=C1    # Benzene
CC1=CC=CC=C1C    # Toluene
C1CCCCC1    # Cyclohexane"""
        self.smiles_input.setText(example_smiles)
    
    def clear_data(self):
        """Clear all data"""
        self.smiles_input.clear()
        self.molecules = []
        self.smiles_list = []
        self.similarity_matrix = None
        
        # Clear heatmap
        self.heatmap_canvas.clear()
        
        # Clear preview
        self.preview_widget.mol1_label.clear()
        self.preview_widget.mol2_label.clear()
        
        # Force update
        self.heatmap_canvas.draw()
        self.preview_widget.update()
    
    def calculate_tonimoto(self):
        """Calculate Tanimoto coefficient between molecules"""
        # Get SMILES strings
        smiles_text = self.smiles_input.toPlainText().strip()
        if not smiles_text:
            QMessageBox.warning(self, "Error", "Please enter at least one SMILES string.")
            return
        
        # Process SMILES strings
        smiles_lines = smiles_text.split('\n')
        self.smiles_list = []
        self.molecules = []
        
        for line in smiles_lines:
            # Remove comments after #
            if '#' in line:
                line = line.split('#')[0].strip()
            
            if not line:
                continue
            
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(line)
            if mol is None:
                QMessageBox.warning(self, "Error", f"Invalid SMILES string: {line}")
                continue
            
            self.smiles_list.append(line)
            self.molecules.append(mol)
        
        if len(self.molecules) < 2:
            QMessageBox.warning(self, "Error", "At least two valid molecules are required.")
            return
        
        # Calculate fingerprints
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in self.molecules]
        
        # Calculate similarity matrix
        n_mols = len(fingerprints)
        self.similarity_matrix = np.zeros((n_mols, n_mols))
        
        for i in range(n_mols):
            for j in range(n_mols):
                self.similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        
        # Display results
        self.display_results()
    
    def display_results(self):
        """Display calculation results"""
        # Display heatmap
        self.heatmap_canvas.plot_heatmap(self.similarity_matrix, self.molecules, self.smiles_list)
    
    def save_results(self):
        """Save results to files"""
        if self.similarity_matrix is None:
            QMessageBox.warning(self, "Error", "No results to save. Please calculate similarities first.")
            return
        
        # Create directory for results
        dir_name = "tonimoto_results"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Save similarity matrix
        matrix_file = os.path.join(dir_name, "similarity_matrix.csv")
        df = pd.DataFrame(self.similarity_matrix,
                         index=[f"Mol {i+1}" for i in range(len(self.molecules))],
                         columns=[f"Mol {i+1}" for i in range(len(self.molecules))])
        df.to_csv(matrix_file)
        
        # Save SMILES strings
        smiles_file = os.path.join(dir_name, "smiles_strings.txt")
        with open(smiles_file, 'w') as f:
            for i, smiles in enumerate(self.smiles_list):
                f.write(f"Mol {i+1}: {smiles}\n")
        
        # Save heatmap plot
        plot_file = os.path.join(dir_name, "similarity_heatmap.png")
        self.heatmap_canvas.save_heatmap(plot_file)
        
        QMessageBox.information(self, "Success", f"Results saved to {dir_name} directory.")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for modern look
    
    # Set default font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    calculator = TonimotoCalculator()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
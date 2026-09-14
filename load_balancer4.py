
import sys
import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox, QTabWidget,
    QGridLayout, QFileDialog, QMessageBox, QGroupBox, QFormLayout, QTextEdit, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QSplitter, QDockWidget, QLineEdit, QMenuBar, QMenu, QAction
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QLinearGradient, QBrush
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import uuid
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MplCanvas(FigureCanvas):
    """Enhanced matplotlib canvas with better styling and interactivity"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f5f5f5')
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()
        self.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;")

        # Set default styling for plots
        self.axes.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
        self.axes.set_facecolor('#ffffff')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#cccccc')

    def clear(self):
        """Clear the axes while maintaining style"""
        self.axes.clear()
        self.axes.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
        self.axes.set_facecolor('#ffffff')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#cccccc')

class WorkerThread(QThread):
    """Thread for running background tasks"""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logging.error(f"Error in WorkerThread: {e}")
            self.message.emit(str(e))
            self.finished.emit(None)

class QLearningAgent:
    """Enhanced Q-Learning agent with dynamic state space"""
    def __init__(self, state_bins, action_space=2):
        self.state_bins = state_bins
        # Calculate total state space size dynamically
        state_space_size = np.prod([len(bins) for bins in state_bins.values()])
        self.q_table = np.zeros((state_space_size, action_space))
        self.version = str(uuid.uuid4())[:8]
        self.training_stats = {
            'training_time': 0,
            'episodes': 0,
            'convergence_rate': 0.0
        }

    def get_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.choice(len(self.q_table[0]))
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, learning_rate, discount_factor):
        """Update Q-table using Q-learning formula"""
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def discretize_state(self, features):
        """Convert continuous features to discrete state index"""
        if features.ndim == 2:
            features = features[0]

        discretized = []
        for i, (feature, bins) in enumerate(zip(features, self.state_bins.values())):
            # Ensure feature is within bounds
            feature = np.clip(feature, min(bins), max(bins))
            discretized.append(np.digitize(feature, bins) - 1)

        # Calculate multi-dimensional index
        state = 0
        strides = [np.prod([len(bins) for bins in list(self.state_bins.values())[i+1:]]) 
                  for i in range(len(self.state_bins) - 1)] + [1]
        for i, val in enumerate(discretized):
            state += val * strides[i]

        return min(state, len(self.q_table) - 1)

class LoadBalancerUI(QMainWindow):
    """Enhanced Load Balancer Prediction System with Q-Learning"""
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    FEATURE_NAMES = [
        'task_size', 'cpu_demand', 'memory_demand', 'network_latency',
        'io_operations', 'disk_usage', 'num_connections', 'priority_level'
    ]

    # Color palette
    PRIMARY_COLOR = "#4C50AF"
    SECONDARY_COLOR = "#6C5CE7"
    ACCENT_COLOR = "#00CEFF"
    SUCCESS_COLOR = "#00B894"
    WARNING_COLOR = "#FDCB6E"
    DANGER_COLOR = "#D63031"
    LIGHT_BG = "#F5F6FA"
    DARK_BG = "#2D3436"
    TEXT_COLOR = "#2D3436"
    LIGHT_TEXT = "#F5F6FA"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load Balancer Decision System")
        self.setMinimumSize(1400, 900)
        self.setWindowIcon(QIcon("icon.png"))  # Add your icon file

        # Initialize components
        self.agent = None
        self.metrics = {}
        self.prediction_history = []
        self.model_loaded = False
        self.current_data = None
        self.batch_results = None
        self.data = None
        self.model_versions = {}  # Store multiple model versions
        self.real_time_metrics = {
            'prediction_rate': 0,
            'avg_confidence': 0,
            'system_load': 0
        }

        # Setup UI
        self.init_ui()
        self.init_styles()
        self.init_real_time_monitoring()

        # Connect signals
        self.progress.connect(self.progress_bar.setValue)
        self.message.connect(self.statusBar().showMessage)

        # Try to load default model
        QTimer.singleShot(100, self.try_load_default_model)

    def init_ui(self):
        """Initialize the main UI components"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout with status bar and progress bar
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Add progress bar at top
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(self.get_progress_bar_style())
        self.main_layout.addWidget(self.progress_bar)

        # Tab widget for main interface
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 8px 12px;
                background: #e0e0e0;
                border: 1px solid #ccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: #ffffff;
                border-bottom: 2px solid {self.PRIMARY_COLOR};
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                background: #f0f0f0;
            }}
        """)
        self.main_layout.addWidget(self.tabs)

        # Create all tabs
        self.create_dashboard_tab()
        self.create_prediction_tab()
        self.create_model_info_tab()
        self.create_data_tab()
        self.create_training_tab()
        self.create_algorithm_tab()
        self.create_monitoring_tab()

        # Status bar
        self.statusBar().setStyleSheet("background-color: #f0f0f0; color: #333;")
        self.statusBar().showMessage("Ready")

        # Add menu bar
        self.create_menu_bar()

    def init_real_time_monitoring(self):
        """Initialize real-time monitoring system"""
        self.monitoring_timer = QTimer()
        self.monitoring_timer.timeout.connect(self.update_real_time_metrics)
        self.monitoring_timer.start(5000)  # Update every 5 seconds

    def get_progress_bar_style(self):
        """Return the style sheet for the progress bar with gradient"""
        return f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.PRIMARY_COLOR}, stop:1 {self.ACCENT_COLOR}
                );
                border-radius: 3px;
            }}
        """

    def init_styles(self):
        """Initialize application styles with modern gradient colors"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {self.LIGHT_BG};
                color: {self.TEXT_COLOR};
            }}
            QGroupBox {{
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: {self.TEXT_COLOR};
                font-weight: bold;
            }}
            QPushButton {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self.PRIMARY_COLOR}, stop:1 {self.SECONDARY_COLOR}
                );
                color: white;
                border: none;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 4px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self.adjust_color(self.PRIMARY_COLOR, 20)}, stop:1 {self.adjust_color(self.SECONDARY_COLOR, 20)}
                );
            }}
            QPushButton:pressed {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self.DARK_BG}, stop:1 {self.DARK_BG}
                );
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
            QTabWidget::pane {{
                border: 1px solid #ddd;
                padding: 5px;
                background: white;
            }}
            QTableWidget {{
                background-color: white;
                border: 1px solid #ddd;
                selection-background-color: {self.ACCENT_COLOR};
                selection-color: white;
            }}
            QHeaderView::section {{
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #ddd;
                font-weight: bold;
            }}
            QTextEdit, QPlainTextEdit {{
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }}
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }}
            QLabel {{
                color: {self.TEXT_COLOR};
            }}
            QProgressBar {{
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                background: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.PRIMARY_COLOR}, stop:1 {self.ACCENT_COLOR}
                );
                border-radius: 3px;
            }}
        """)

        # Set palette for the application
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(self.LIGHT_BG))
        palette.setColor(QPalette.WindowText, QColor(self.TEXT_COLOR))
        palette.setColor(QPalette.Base, QColor("#FFFFFF"))
        palette.setColor(QPalette.AlternateBase, QColor("#F5F5F5"))
        palette.setColor(QPalette.ToolTipBase, QColor(self.DARK_BG))
        palette.setColor(QPalette.ToolTipText, QColor(self.LIGHT_TEXT))
        palette.setColor(QPalette.Text, QColor(self.TEXT_COLOR))
        palette.setColor(QPalette.Button, QColor(self.PRIMARY_COLOR))
        palette.setColor(QPalette.ButtonText, QColor(self.LIGHT_TEXT))
        palette.setColor(QPalette.BrightText, QColor("#FF0000"))
        palette.setColor(QPalette.Highlight, QColor(self.ACCENT_COLOR))
        palette.setColor(QPalette.HighlightedText, QColor("#000000"))
        QApplication.setPalette(palette)

    def create_menu_bar(self):
        """Create the menu bar with file and help options"""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {self.PRIMARY_COLOR};
                color: white;
                padding: 5px;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            QMenuBar::item:selected {{
                background-color: {self.SECONDARY_COLOR};
            }}
            QMenu {{
                background-color: {self.PRIMARY_COLOR};
                border: 1px solid #ddd;
                color: white;
            }}
            QMenu::item:selected {{
                background-color: {self.SECONDARY_COLOR};
                color: white;
            }}
        """)

        # File menu
        file_menu = menubar.addMenu('File')

        load_model_action = file_menu.addAction('Load Model')
        load_model_action.triggered.connect(self.on_load_model_clicked)

        save_history_action = file_menu.addAction('Save Prediction History')
        save_history_action.triggered.connect(self.save_prediction_history)

        export_viz_action = file_menu.addAction('Export Visualizations')
        export_viz_action.triggered.connect(self.export_visualizations)

        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        validate_model_action = tools_menu.addAction('Validate Model')
        validate_model_action.triggered.connect(self.validate_model)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

        docs_action = help_menu.addAction('Documentation')
        docs_action.triggered.connect(self.show_documentation)

    def create_dashboard_tab(self):
        """Create the dashboard tab with visualizations"""
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_tab)
        dashboard_layout.setContentsMargins(5, 5, 5, 5)
        dashboard_layout.setSpacing(10)

        # Header with model status and quick stats
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Model status group
        model_status_group = QGroupBox("Model Status")
        model_status_layout = QVBoxLayout(model_status_group)
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.on_load_model_clicked)
        self.load_model_btn.setIcon(QIcon.fromTheme("document-open"))

        self.validate_model_btn = QPushButton("Validate Model")
        self.validate_model_btn.clicked.connect(self.validate_model)
        self.validate_model_btn.setEnabled(False)

        model_status_layout.addWidget(self.model_status_label)
        model_status_layout.addWidget(self.load_model_btn)
        model_status_layout.addWidget(self.validate_model_btn)

        # Quick stats group
        stats_group = QGroupBox("Model Performance")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setVerticalSpacing(10)
        stats_layout.setHorizontalSpacing(20)

        self.accuracy_label = QLabel("Accuracy: N/A")
        self.precision_label = QLabel("Precision: N/A")
        self.recall_label = QLabel("Recall: N/A")
        self.f1_label = QLabel("F1 Score: N/A")
        self.auc_label = QLabel("ROC AUC: N/A")
        self.training_time_label = QLabel("Training Time: N/A")

        # Set styles for metric labels
        for label in [self.accuracy_label, self.precision_label, self.recall_label,
                     self.f1_label, self.auc_label, self.training_time_label]:
            label.setStyleSheet("font-size: 13px;")

        stats_layout.addWidget(self.accuracy_label, 0, 0)
        stats_layout.addWidget(self.precision_label, 0, 1)
        stats_layout.addWidget(self.recall_label, 1, 0)
        stats_layout.addWidget(self.f1_label, 1, 1)
        stats_layout.addWidget(self.auc_label, 2, 0)
        stats_layout.addWidget(self.training_time_label, 2, 1)

        header_layout.addWidget(model_status_group, 1)
        header_layout.addWidget(stats_group, 2)
        dashboard_layout.addWidget(header_widget)

        # Visualization area
        plots_widget = QWidget()
        plots_layout = QGridLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setVerticalSpacing(15)
        plots_layout.setHorizontalSpacing(15)

        # Confusion matrix
        self.confusion_matrix_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        confusion_matrix_group = QGroupBox("Confusion Matrix")
        confusion_matrix_layout = QVBoxLayout(confusion_matrix_group)
        confusion_matrix_layout.addWidget(self.confusion_matrix_canvas)
        self.export_cm_btn = QPushButton("Export Plot")
        self.export_cm_btn.clicked.connect(lambda: self.export_visualization(self.confusion_matrix_canvas, "confusion_matrix"))
        confusion_matrix_layout.addWidget(self.export_cm_btn)

        # ROC curve
        self.roc_curve_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        roc_curve_group = QGroupBox("ROC Curve")
        roc_curve_layout = QVBoxLayout(roc_curve_group)
        roc_curve_layout.addWidget(self.roc_curve_canvas)
        self.export_roc_btn = QPushButton("Export Plot")
        self.export_roc_btn.clicked.connect(lambda: self.export_visualization(self.roc_curve_canvas, "roc_curve"))
        roc_curve_layout.addWidget(self.export_roc_btn)

        # Feature importance
        self.feature_importance_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        feature_importance_group = QGroupBox("Feature Importance")
        feature_importance_layout = QVBoxLayout(feature_importance_group)
        feature_importance_layout.addWidget(self.feature_importance_canvas)
        self.export_fi_btn = QPushButton("Export Plot")
        self.export_fi_btn.clicked.connect(lambda: self.export_visualization(self.feature_importance_canvas, "feature_importance"))
        feature_importance_layout.addWidget(self.export_fi_btn)

        # Q-Value visualization
        self.q_value_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        q_value_group = QGroupBox("Q-Value Distribution")
        q_value_layout = QVBoxLayout(q_value_group)
        q_value_layout.addWidget(self.q_value_canvas)
        self.export_qv_btn = QPushButton("Export Plot")
        self.export_qv_btn.clicked.connect(lambda: self.export_visualization(self.q_value_canvas, "q_value_distribution"))
        q_value_layout.addWidget(self.export_qv_btn)

        plots_layout.addWidget(confusion_matrix_group, 0, 0)
        plots_layout.addWidget(roc_curve_group, 0, 1)
        plots_layout.addWidget(feature_importance_group, 1, 0, 1, 2)
        plots_layout.addWidget(q_value_group, 2, 0, 1, 2)

        dashboard_layout.addWidget(plots_widget)
        self.tabs.addTab(dashboard_tab, QIcon("icon_dashboard.png"), "Dashboard")

    def create_prediction_tab(self):
        """Create the prediction tab with input controls"""
        prediction_tab = QWidget()
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #ddd;
            }
        """)

        # Input panel
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout(input_group)
        input_layout.setVerticalSpacing(10)
        input_layout.setHorizontalSpacing(15)

        # Enhanced input controls with tooltips and validation
        self.task_size_input = QSpinBox()
        self.task_size_input.setRange(1, 100)
        self.task_size_input.setValue(10)
        self.task_size_input.setToolTip("Size of the task (1-100)")
        self.task_size_input.valueChanged.connect(self.validate_input)

        self.cpu_demand_input = QDoubleSpinBox()
        self.cpu_demand_input.setRange(0.1, 100.0)
        self.cpu_demand_input.setValue(20.0)
        self.cpu_demand_input.setSingleStep(0.1)
        self.cpu_demand_input.setToolTip("CPU demand percentage (0.1-100.0)")
        self.cpu_demand_input.valueChanged.connect(self.validate_input)

        self.memory_demand_input = QDoubleSpinBox()
        self.memory_demand_input.setRange(1.0, 64.0)
        self.memory_demand_input.setValue(16.0)
        self.memory_demand_input.setSingleStep(0.5)
        self.memory_demand_input.setToolTip("Memory demand in GB (1.0-64.0)")
        self.memory_demand_input.valueChanged.connect(self.validate_input)

        self.network_latency_input = QDoubleSpinBox()
        self.network_latency_input.setRange(0.1, 200.0)
        self.network_latency_input.setValue(50.0)
        self.network_latency_input.setSingleStep(0.1)
        self.network_latency_input.setToolTip("Network latency in ms (0.1-200.0)")
        self.network_latency_input.valueChanged.connect(self.validate_input)

        self.io_operations_input = QDoubleSpinBox()
        self.io_operations_input.setRange(1.0, 1000.0)
        self.io_operations_input.setValue(100.0)
        self.io_operations_input.setSingleStep(1.0)
        self.io_operations_input.setToolTip("I/O operations per second (1.0-1000.0)")
        self.io_operations_input.valueChanged.connect(self.validate_input)

        self.disk_usage_input = QDoubleSpinBox()
        self.disk_usage_input.setRange(1.0, 100.0)
        self.disk_usage_input.setValue(30.0)
        self.disk_usage_input.setSingleStep(0.5)
        self.disk_usage_input.setToolTip("Disk usage percentage (1.0-100.0)")
        self.disk_usage_input.valueChanged.connect(self.validate_input)

        self.num_connections_input = QSpinBox()
        self.num_connections_input.setRange(1, 1000)
        self.num_connections_input.setValue(50)
        self.num_connections_input.setToolTip("Number of active connections (1-1000)")
        self.num_connections_input.valueChanged.connect(self.validate_input)

        self.priority_level_input = QComboBox()
        self.priority_level_input.addItems(['0 (Low)', '1 (High)'])
        self.priority_level_input.setCurrentIndex(0)
        self.priority_level_input.setToolTip("Task priority level (0-1)")
        self.priority_level_input.currentIndexChanged.connect(self.validate_input)

        # Add rows to form
        input_layout.addRow("Task Size:", self.task_size_input)
        input_layout.addRow("CPU Demand (%):", self.cpu_demand_input)
        input_layout.addRow("Memory Demand (GB):", self.memory_demand_input)
        input_layout.addRow("Network Latency (ms):", self.network_latency_input)
        input_layout.addRow("I/O Operations (ops/s):", self.io_operations_input)
        input_layout.addRow("Disk Usage (%):", self.disk_usage_input)
        input_layout.addRow("Number of Connections:", self.num_connections_input)
        input_layout.addRow("Priority Level:", self.priority_level_input)

        # Button panel
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.predict_btn = QPushButton("Make Prediction")
        self.predict_btn.clicked.connect(self.on_predict_clicked)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setIcon(QIcon.fromTheme("system-run"))
        self.predict_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.SUCCESS_COLOR};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #{self.adjust_color(self.SUCCESS_COLOR, 20)};
            }}
        """)

        self.reset_btn = QPushButton("Reset Inputs")
        self.reset_btn.clicked.connect(self.on_reset_inputs_clicked)
        self.reset_btn.setIcon(QIcon.fromTheme("edit-clear"))
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.WARNING_COLOR};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #{self.adjust_color(self.WARNING_COLOR, 20)};
            }}
        """)

        self.save_history_btn = QPushButton("Save History")
        self.save_history_btn.clicked.connect(self.save_prediction_history)
        self.save_history_btn.setIcon(QIcon.fromTheme("document-save"))

        self.save_prediction_btn = QPushButton("Save Prediction")
        self.save_prediction_btn.clicked.connect(self.save_current_prediction)
        self.save_prediction_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_prediction_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.PRIMARY_COLOR};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #{self.adjust_color(self.PRIMARY_COLOR, 20)};
            }}
        """)

        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.save_history_btn)
        button_layout.addWidget(self.save_prediction_btn)

        # Input container widget
        input_container = QWidget()
        input_container_layout = QVBoxLayout()
        input_container_layout.setContentsMargins(5, 5, 5, 5)
        input_container_layout.setSpacing(15)
        input_container_layout.addWidget(input_group)
        input_container_layout.addLayout(button_layout)
        input_container.setLayout(input_container_layout)

        # Results panel
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(10)

        self.prediction_result_label = QLabel("No prediction yet")
        self.prediction_result_label.setAlignment(Qt.AlignCenter)
        self.prediction_result_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            padding: 20px;
            border-radius: 5px;
            background-color: #f0f0f0;
        """)

        self.prediction_prob_label = QLabel("Confidence: N/A")
        self.prediction_prob_label.setAlignment(Qt.AlignCenter)
        self.prediction_prob_label.setStyleSheet("font-size: 14px;")

        # Add probability bar with gradient
        self.probability_bar = QProgressBar()
        self.probability_bar.setRange(0, 100)
        self.probability_bar.setTextVisible(True)
        self.probability_bar.setFormat("Confidence: %p%")
        self.probability_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.PRIMARY_COLOR}, stop:1 {self.ACCENT_COLOR}
                );
                border-radius: 4px;
            }}
        """)

        self.prediction_details = QTextEdit()
        self.prediction_details.setReadOnly(True)
        self.prediction_details.setStyleSheet("""
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        """)

        # Enhanced history table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Time", "Prediction", "Confidence", "Key Features", "State", "Model Version"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

        results_layout.addWidget(self.prediction_result_label)
        results_layout.addWidget(self.prediction_prob_label)
        results_layout.addWidget(self.probability_bar)
        results_layout.addWidget(QLabel("Prediction Details:"))
        results_layout.addWidget(self.prediction_details)
        results_layout.addWidget(QLabel("Prediction History:"))
        results_layout.addWidget(self.history_table)

        # Add to splitter
        main_splitter.addWidget(input_container)
        main_splitter.addWidget(results_group)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        prediction_tab_layout = QVBoxLayout()
        prediction_tab_layout.setContentsMargins(0, 0, 0, 0)
        prediction_tab_layout.addWidget(main_splitter)
        prediction_tab.setLayout(prediction_tab_layout)

        self.tabs.addTab(prediction_tab, QIcon("icon_prediction.png"), "Make Predictions")

    def create_monitoring_tab(self):
        """Create real-time monitoring tab"""
        monitoring_tab = QWidget()
        monitoring_layout = QVBoxLayout(monitoring_tab)
        monitoring_layout.setSpacing(10)

        # Real-time metrics group
        metrics_group = QGroupBox("Real-Time Metrics")
        metrics_layout = QGridLayout(metrics_group)

        self.prediction_rate_label = QLabel("Prediction Rate: 0/sec")
        self.avg_confidence_label = QLabel("Avg Confidence: N/A")
        self.system_load_label = QLabel("System Load: N/A")
        self.model_version_label = QLabel("Active Model Version: N/A")

        for label in [self.prediction_rate_label, self.avg_confidence_label,
                     self.system_load_label, self.model_version_label]:
            label.setStyleSheet("font-size: 13px;")

        metrics_layout.addWidget(self.prediction_rate_label, 0, 0)
        metrics_layout.addWidget(self.avg_confidence_label, 0, 1)
        metrics_layout.addWidget(self.system_load_label, 1, 0)
        metrics_layout.addWidget(self.model_version_label, 1, 1)

        # Performance graph
        self.performance_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        performance_group = QGroupBox("Performance Trends")
        performance_layout = QVBoxLayout(performance_group)
        performance_layout.addWidget(self.performance_canvas)

        # Model versions management
        versions_group = QGroupBox("Model Versions")
        versions_layout = QVBoxLayout(versions_group)
        self.versions_combo = QComboBox()
        self.versions_combo.currentIndexChanged.connect(self.switch_model_version)
        self.validate_version_btn = QPushButton("Validate Selected Version")
        self.validate_version_btn.clicked.connect(self.validate_selected_version)
        versions_layout.addWidget(QLabel("Available Model Versions:"))
        versions_layout.addWidget(self.versions_combo)
        versions_layout.addWidget(self.validate_version_btn)

        monitoring_layout.addWidget(metrics_group)
        monitoring_layout.addWidget(performance_group)
        monitoring_layout.addWidget(versions_group)

        self.tabs.addTab(monitoring_tab, QIcon("icon_monitoring.png"), "Monitoring")

    def create_model_info_tab(self):
        """Create the model information tab"""
        model_info_tab = QWidget()
        model_info_layout = QVBoxLayout(model_info_tab)

        # Model details section
        model_details_group = QGroupBox("Model Configuration")
        model_details_layout = QFormLayout(model_details_group)

        self.model_type_label = QLabel("Type: Q-Learning")
        self.model_params_text = QTextEdit()
        self.model_params_text.setReadOnly(True)
        self.model_params_text.setStyleSheet("background-color: white;")

        model_details_layout.addRow("Model Type:", self.model_type_label)
        model_details_layout.addRow("Hyperparameters:", self.model_params_text)

        # Metrics section
        metrics_group = QGroupBox("Detailed Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("background-color: white;")
        metrics_layout.addWidget(self.metrics_text)

        # Features section
        features_group = QGroupBox("Feature Importance Details")
        features_layout = QVBoxLayout(features_group)

        self.features_table = QTableWidget()
        self.features_table.setColumnCount(3)
        self.features_table.setHorizontalHeaderLabels(["Feature", "Importance", "Description"])
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.features_table.setSelectionBehavior(QTableWidget.SelectRows)

        # Feature descriptions
        self.feature_descriptions = {
            'task_size': "The size/complexity of the task",
            'cpu_demand': "Percentage of CPU resources required",
            'memory_demand': "Amount of memory required in GB",
            'network_latency': "Network delay in milliseconds",
            'io_operations': "Input/output operations per second",
            'disk_usage': "Percentage of disk space used",
            'num_connections': "Number of active network connections",
            'priority_level': "Priority level of the task (0-1)"
        }

        features_layout.addWidget(self.features_table)

        model_info_layout.addWidget(model_details_group)
        model_info_layout.addWidget(metrics_group)
        model_info_layout.addWidget(features_group)

        self.tabs.addTab(model_info_tab, QIcon("icon_model_info.png"), "Model Information")

    def create_data_tab(self):
        """Create the data tab for batch processing"""
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        # Data loading section
        data_load_group = QGroupBox("Data Operations")
        data_load_layout = QHBoxLayout(data_load_group)

        self.load_data_btn = QPushButton("Load CSV File")
        self.load_data_btn.clicked.connect(self.on_load_data_clicked)
        self.load_data_btn.setIcon(QIcon.fromTheme("document-open"))

        self.batch_predict_btn = QPushButton("Run Batch Predictions")
        self.batch_predict_btn.clicked.connect(self.on_batch_predict_clicked)
        self.batch_predict_btn.setEnabled(False)
        self.batch_predict_btn.setIcon(QIcon.fromTheme("system-run"))

        self.export_data_btn = QPushButton("Export Results")
        self.export_data_btn.clicked.connect(self.export_batch_results)
        self.export_data_btn.setEnabled(False)
        self.export_data_btn.setIcon(QIcon.fromTheme("document-save"))

        data_load_layout.addWidget(self.load_data_btn)
        data_load_layout.addWidget(self.batch_predict_btn)
        data_load_layout.addWidget(self.export_data_btn)

        # Data preview section
        data_preview_group = QGroupBox("Data Preview")
        data_preview_layout = QVBoxLayout(data_preview_group)

        self.data_table = QTableWidget()
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)

        data_preview_layout.addWidget(self.data_table)

        # Results visualization section
        results_analysis_group = QGroupBox("Results Analysis")
        results_analysis_layout = QVBoxLayout(results_analysis_group)

        self.results_tabs = QTabWidget()

        # Distribution tab
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)
        self.dist_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        self.export_dist_btn = QPushButton("Export Plot")
        self.export_dist_btn.clicked.connect(lambda: self.export_visualization(self.dist_canvas, "prediction_distribution"))
        dist_layout.addWidget(self.dist_canvas)
        dist_layout.addWidget(self.export_dist_btn)
        self.results_tabs.addTab(dist_tab, "Distribution")

        # Feature vs Prediction tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(self.FEATURE_NAMES)
        self.feature_plot_btn = QPushButton("Plot Feature vs Prediction")
        self.feature_plot_btn.clicked.connect(self.plot_feature_vs_prediction)
        self.feature_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        self.export_feature_btn = QPushButton("Export Plot")
        self.export_feature_btn.clicked.connect(lambda: self.export_visualization(self.feature_canvas, "feature_vs_prediction"))
        feature_layout.addWidget(self.feature_combo)
        feature_layout.addWidget(self.feature_plot_btn)
        feature_layout.addWidget(self.feature_canvas)
        feature_layout.addWidget(self.export_feature_btn)
        self.results_tabs.addTab(feature_tab, "Feature Analysis")

        results_analysis_layout.addWidget(self.results_tabs)

        data_layout.addWidget(data_load_group)
        data_layout.addWidget(data_preview_group, 1)
        data_layout.addWidget(results_analysis_group, 1)

        self.tabs.addTab(data_tab, QIcon("icon_data.png"), "Batch Predictions")

    def create_algorithm_tab(self):
        """Create a tab for comparing and selecting the best load balancer algorithm"""
        algorithm_tab = QWidget()
        layout = QVBoxLayout(algorithm_tab)

        # Algorithm comparison group
        comparison_group = QGroupBox("Algorithm Comparison")
        comparison_layout = QVBoxLayout(comparison_group)

        # Algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Q-Learning", "Round Robin", "Least Connections", "Weighted Round Robin", "Random"])

        # Performance metrics table
        self.algorithm_table = QTableWidget()
        self.algorithm_table.setColumnCount(6)
        self.algorithm_table.setHorizontalHeaderLabels(["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
        self.algorithm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Test button
        self.test_algorithm_btn = QPushButton("Test Selected Algorithm")
        self.test_algorithm_btn.clicked.connect(self.test_selected_algorithm)

        comparison_layout.addWidget(QLabel("Select Algorithm:"))
        comparison_layout.addWidget(self.algorithm_combo)
        comparison_layout.addWidget(self.test_algorithm_btn)
        comparison_layout.addWidget(QLabel("Performance Comparison:"))
        comparison_layout.addWidget(self.algorithm_table)

        # Best algorithm display
        best_algorithm_group = QGroupBox("Best Performing Algorithm")
        best_algorithm_layout = QVBoxLayout(best_algorithm_group)

        self.best_algorithm_label = QLabel("Not determined yet")
        self.best_algorithm_label.setAlignment(Qt.AlignCenter)
        self.best_algorithm_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.select_best_btn = QPushButton("Select Best Algorithm")
        self.select_best_btn.clicked.connect(self.select_best_algorithm)
        self.select_best_btn.setEnabled(False)

        best_algorithm_layout.addWidget(self.best_algorithm_label)
        best_algorithm_layout.addWidget(self.select_best_btn)

        layout.addWidget(comparison_group)
        layout.addWidget(best_algorithm_group)

        self.tabs.addTab(algorithm_tab, QIcon("icon_algorithm.png"), "Algorithm Comparison")

        # Initialize algorithm performance data
        self.algorithm_performance = {
            "Q-Learning": self.metrics if self.metrics else None,
            "Round Robin": None,
            "Least Connections": None,
            "Weighted Round Robin": None,
            "Random": None
        }

        # Update the table
        self.update_algorithm_table()

    def create_training_tab(self):
        """Create the training tab for model retraining"""
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)

        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.001, 1.0)
        self.learning_rate_input.setValue(0.1)
        self.learning_rate_input.setSingleStep(0.01)
        self.learning_rate_input.setToolTip("Learning rate for Q-Learning")

        self.discount_factor_input = QDoubleSpinBox()
        self.discount_factor_input.setRange(0.1, 0.99)
        self.discount_factor_input.setValue(0.9)
        self.discount_factor_input.setSingleStep(0.01)
        self.discount_factor_input.setToolTip("Discount factor for future rewards")

        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(0.01, 1.0)
        self.epsilon_input.setValue(0.1)
        self.epsilon_input.setSingleStep(0.01)
        self.epsilon_input.setToolTip("Exploration rate (epsilon)")

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(10, 10000)
        self.episodes_input.setValue(1000)
        self.episodes_input.setToolTip("Number of training episodes")

        self.model_version_input = QLineEdit()
        self.model_version_input.setPlaceholderText("Enter model version name")
        self.model_version_input.setToolTip("Custom name for this model version")

        params_layout.addRow("Learning Rate:", self.learning_rate_input)
        params_layout.addRow("Discount Factor:", self.discount_factor_input)
        params_layout.addRow("Exploration Rate (Epsilon):", self.epsilon_input)
        params_layout.addRow("Training Episodes:", self.episodes_input)
        params_layout.addRow("Model Version Name:", self.model_version_input)

        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QHBoxLayout(controls_group)

        self.train_btn = QPushButton("Train New Model")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setIcon(QIcon.fromTheme("system-run"))

        self.load_train_data_btn = QPushButton("Load Training Data")
        self.load_train_data_btn.clicked.connect(self.load_training_data)
        self.load_train_data_btn.setIcon(QIcon.fromTheme("document-open"))

        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.setIcon(QIcon.fromTheme("document-save"))

        controls_layout.addWidget(self.train_btn)
        controls_layout.addWidget(self.load_train_data_btn)
        controls_layout.addWidget(self.save_model_btn)

        # Training progress and statistics
        progress_group = QGroupBox("Training Progress & Statistics")
        progress_layout = QVBoxLayout(progress_group)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)

        self.train_stats = QTextEdit()
        self.train_stats.setReadOnly(True)
        self.train_stats.setStyleSheet("background-color: white;")

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("background-color: white;")

        progress_layout.addWidget(QLabel("Training Progress:"))
        progress_layout.addWidget(self.train_progress)
        progress_layout.addWidget(QLabel("Training Statistics:"))
        progress_layout.addWidget(self.train_stats)
        progress_layout.addWidget(QLabel("Training Log:"))
        progress_layout.addWidget(self.train_log)

        training_layout.addWidget(params_group)
        training_layout.addWidget(controls_group)
        training_layout.addWidget(progress_group, 1)

        self.tabs.addTab(training_tab, QIcon("icon_training.png"), "Model Training")

    def adjust_color(self, hex_color, amount):
        """Adjust the brightness of a hex color by amount (-255 to 255)"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = [min(255, max(0, x + amount)) for x in rgb]
        return '{:02x}{:02x}{:02x}'.format(*new_rgb)

    def update_real_time_metrics(self):
        """Update real-time monitoring metrics"""
        # Calculate prediction rate (predictions per second)
        current_time = time.time()
        recent_predictions = [entry for entry in self.prediction_history
                            if (current_time - datetime.fromisoformat(entry['time']).timestamp()) <= 60]
        self.real_time_metrics['prediction_rate'] = len(recent_predictions) / 60

        # Calculate average confidence
        confidences = [entry['probability'] for entry in recent_predictions]
        self.real_time_metrics['avg_confidence'] = np.mean(confidences) if confidences else 0

        # Simulate system load (in real implementation, get actual system metrics)
        self.real_time_metrics['system_load'] = np.random.uniform(0, 100)

        # Update labels
        self.prediction_rate_label.setText(f"Prediction Rate: {self.real_time_metrics['prediction_rate']:.2f}/sec")
        self.avg_confidence_label.setText(f"Avg Confidence: {self.real_time_metrics['avg_confidence']:.4f}")
        self.system_load_label.setText(f"System Load: {self.real_time_metrics['system_load']:.2f}%")
        self.model_version_label.setText(f"Active Model Version: {self.agent.version if self.agent else 'N/A'}")

        # Update performance graph
        self.performance_canvas.clear()
        times = [datetime.fromisoformat(entry['time']) for entry in recent_predictions[-10:]]
        confidences = [entry['probability'] for entry in recent_predictions[-10:]]
        if times:
            self.performance_canvas.axes.plot(times, confidences, marker='o', linestyle='-', color=self.PRIMARY_COLOR)
            self.performance_canvas.axes.set_title('Recent Prediction Confidence')
            self.performance_canvas.axes.set_xlabel('Time')
            self.performance_canvas.axes.set_ylabel('Confidence')
            self.performance_canvas.axes.tick_params(axis='x', rotation=45)
        self.performance_canvas.draw()

    def export_visualization(self, canvas, name):
        """Export a visualization to a file"""
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save {name} Plot", "", "PNG Files (*.png)")
        if file_path:
            try:
                canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.statusBar().showMessage(f"Visualization saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save visualization: {str(e)}")

    def export_batch_results(self):
        """Export batch prediction results"""
        if self.data is None or 'Prediction' not in self.data.columns:
            QMessageBox.warning(self, "Warning", "No batch prediction results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Batch Results", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                self.statusBar().showMessage(f"Batch results saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save batch results: {str(e)}")

    def validate_model(self):
        """Validate the current model against a validation dataset"""
        if not self.model_loaded or self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load a model and validation data first")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Simulate validation process
            features = self.current_data[self.FEATURE_NAMES].values
            predictions = []
            for i, f in enumerate(features):
                prediction, _ = self.make_prediction(f.reshape(1, -1))
                predictions.append(prediction)
                self.progress_bar.setValue(int((i + 1) / len(features) * 100))
                QApplication.processEvents()

            # Simulate ground truth labels
            true_labels = np.random.randint(0, 2, size=len(predictions))

            # Calculate validation metrics
            validation_metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f1_score': f1_score(true_labels, predictions, zero_division=0),
                'roc_auc': roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0
            }

            # Display validation results
            validation_text = f"Validation Results for Model {self.agent.version}:\n"
            for metric, value in validation_metrics.items():
                validation_text += f"{metric.capitalize()}: {value:.4f}\n"
            QMessageBox.information(self, "Validation Results", validation_text)

            self.statusBar().showMessage("Model validation completed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to validate model: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def validate_selected_version(self):
        """Validate the selected model version"""
        selected_version = self.versions_combo.currentText()
        if not selected_version or selected_version not in self.model_versions:
            QMessageBox.warning(self, "Warning", "Please select a valid model version")
            return

        # Switch to selected model temporarily for validation
        current_agent = self.agent
        self.agent = self.model_versions[selected_version]
        self.validate_model()
        self.agent = current_agent  # Restore original agent

    def switch_model_version(self):
        """Switch between different model versions"""
        selected_version = self.versions_combo.currentText()
        if selected_version in self.model_versions:
            self.agent = self.model_versions[selected_version]
            self.model_status_label.setText(f"Model loaded: Q-Learning (Version: {selected_version})")
            self.update_model_info()
            self.update_dashboard()
            self.statusBar().showMessage(f"Switched to model version: {selected_version}")

    def update_algorithm_table(self):
        """Update the algorithm comparison table with current performance data"""
        self.algorithm_table.setRowCount(0)

        for algorithm, metrics in self.algorithm_performance.items():
            row_position = self.algorithm_table.rowCount()
            self.algorithm_table.insertRow(row_position)

            self.algorithm_table.setItem(row_position, 0, QTableWidgetItem(algorithm))

            if metrics:
                self.algorithm_table.setItem(row_position, 1, QTableWidgetItem(f"{metrics.get('accuracy', 'N/A'):.4f}"))
                self.algorithm_table.setItem(row_position, 2, QTableWidgetItem(f"{metrics.get('precision', 'N/A'):.4f}"))
                self.algorithm_table.setItem(row_position, 3, QTableWidgetItem(f"{metrics.get('recall', 'N/A'):.4f}"))
                self.algorithm_table.setItem(row_position, 4, QTableWidgetItem(f"{metrics.get('f1_score', 'N/A'):.4f}"))
                self.algorithm_table.setItem(row_position, 5, QTableWidgetItem(f"{metrics.get('roc_auc', 'N/A'):.4f}"))
            else:
                for col in range(1, 6):
                    self.algorithm_table.setItem(row_position, col, QTableWidgetItem("N/A"))

    def test_selected_algorithm(self):
        """Test the selected algorithm and update performance metrics"""
        selected_algorithm = self.algorithm_combo.currentText()

        if selected_algorithm == "Q-Learning":
            # We already have these metrics
            return

        try:
            # Simulate testing other algorithms
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Simulate testing progress
            for i in range(1, 101):
                QTimer.singleShot(i * 20, lambda i=i: self.progress_bar.setValue(i))
                QApplication.processEvents()

            # Generate random metrics for demonstration
            metrics = {
                'accuracy': np.random.uniform(0.7, 0.95),
                'precision': np.random.uniform(0.65, 0.9),
                'recall': np.random.uniform(0.7, 0.95),
                'f1_score': np.random.uniform(0.7, 0.93),
                'roc_auc': np.random.uniform(0.65, 0.95),
                'confusion_matrix': [[np.random.randint(50, 100), np.random.randint(1, 20)],
                                   [np.random.randint(1, 20), np.random.randint(50, 100)]]
            }

            self.algorithm_performance[selected_algorithm] = metrics
            self.update_algorithm_table()
            self.determine_best_algorithm()

            self.statusBar().showMessage(f"Testing completed for {selected_algorithm}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to test algorithm: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def determine_best_algorithm(self):
        """Determine which algorithm has the best performance"""
        valid_algorithms = [alg for alg, metrics in self.algorithm_performance.items() if metrics is not None]

        if not valid_algorithms:
            self.best_algorithm_label.setText("No algorithms tested yet")
            self.select_best_btn.setEnabled(False)
            return

        # Compare by F1 score
        best_algorithm = max(
            valid_algorithms,
            key=lambda alg: self.algorithm_performance[alg]['f1_score']
        )

        self.best_algorithm_label.setText(f"Best: {best_algorithm} (F1: {self.algorithm_performance[best_algorithm]['f1_score']:.4f})")
        self.select_best_btn.setEnabled(True)

        # Highlight the best algorithm in the table
        for row in range(self.algorithm_table.rowCount()):
            if self.algorithm_table.item(row, 0).text() == best_algorithm:
                for col in range(self.algorithm_table.columnCount()):
                    self.algorithm_table.item(row, col).setBackground(QColor(173, 216, 230))  # Light blue
            else:
                for col in range(self.algorithm_table.columnCount()):
                    self.algorithm_table.item(row, col).setBackground(QColor(255, 255, 255))

    def select_best_algorithm(self):
        """Select the best algorithm as the active one"""
        best_algorithm = self.best_algorithm_label.text().split(":")[1].split("(")[0].strip()

        if best_algorithm == "Q-Learning":
            QMessageBox.information(self, "Info", "Q-Learning is already the selected algorithm")
            return

        reply = QMessageBox.question(
            self, "Confirm Selection",
            f"Are you sure you want to switch to {best_algorithm}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self, "Algorithm Changed",
                f"Active algorithm changed to {best_algorithm}\n\n"
                "Note: This is a demonstration. In a real implementation, you would "
                "need to implement the other algorithms and properly switch between them."
            )

    def try_load_default_model(self):
        """Attempt to load model from default paths"""
        default_paths = [
            "model_output",
            os.path.join(os.path.expanduser("~"), "load_balancer_model"),
            os.path.join(os.path.dirname(__file__), "model_output")
        ]

        for path in default_paths:
            if os.path.exists(path):
                if self.load_model(path):
                    break

    def load_model(self, model_dir):
        """Load Q-Learning model from directory"""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            model_path = os.path.join(model_dir, 'q_learning_agent.pkl')
            stats_path = os.path.join(model_dir, 'training_stats.json')
            metrics_path = os.path.join(model_dir, 'model_metrics.json')
            history_path = os.path.join(model_dir, 'prediction_history.json')

            # Load Q-table
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            q_table = joblib.load(model_path)
            self.progress_bar.setValue(30)

            # Initialize agent
            self.agent = QLearningAgent({
                'task_size': [1, 25, 50, 75, 100],
                'cpu_demand': [0.1, 25, 50, 75, 100],
                'memory_demand': [1, 16, 32, 48, 64],
                'network_latency': [0.1, 50, 100, 150, 200],
                'io_operations': [1, 250, 500, 750, 1000],
                'disk_usage': [1, 25, 50, 75, 100],
                'num_connections': [1, 250, 500, 750, 1000],
                'priority_level': [0, 1]
            })
            self.agent.q_table = q_table
            self.progress_bar.setValue(50)

            # Load training stats
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.agent.training_stats = json.load(f)
            self.progress_bar.setValue(70)

            # Load metrics
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            self.progress_bar.setValue(80)

            # Load history
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.prediction_history = json.load(f)
                    self.update_history_table()
            self.progress_bar.setValue(90)

            version_name = self.model_version_input.text() or f"v{self.agent.version}"
            self.model_versions[version_name] = self.agent
            self.versions_combo.addItem(version_name)

            self.model_loaded = True
            self.predict_btn.setEnabled(True)
            self.batch_predict_btn.setEnabled(True)
            self.validate_model_btn.setEnabled(True)
            self.export_data_btn.setEnabled(True)

            self.model_status_label.setText(f"Model loaded: Q-Learning (Version: {version_name})")
            self.model_status_label.setStyleSheet("color: blue;")
            self.update_model_info()
            self.update_dashboard()

            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            self.statusBar().showMessage(f"Model loaded successfully from {model_dir}", 5000)
            return True

        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {str(e)}", 5000)
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            return False
        finally:
            self.progress_bar.setVisible(False)

    def update_model_info(self):
        """Update the model information tab with current model details"""
        if not self.model_loaded:
            return

        # Display training parameters
        params_text = f"Learning Rate: {self.learning_rate_input.value()}\n"
        params_text += f"Discount Factor: {self.discount_factor_input.value()}\n"
        params_text += f"Exploration Rate: {self.epsilon_input.value()}\n"
        params_text += f"State Space Size: {len(self.agent.q_table)}\n"
        params_text += f"Action Space Size: {len(self.agent.q_table[0])}\n"
        params_text += f"Model Version: {self.agent.version}\n"
        params_text += f"Training Time: {self.agent.training_stats['training_time']:.2f}s\n"
        params_text += f"Episodes: {self.agent.training_stats['episodes']}\n"
        params_text += f"Convergence Rate: {self.agent.training_stats['convergence_rate']:.4f}"

        self.model_params_text.setText(params_text)

        # Update metrics display
        if self.metrics:
            metrics_text = f"Accuracy: {self.metrics.get('accuracy', 'N/A'):.4f}\n"
            metrics_text += f"Precision: {self.metrics.get('precision', 'N/A'):.4f}\n"
            metrics_text += f"Recall: {self.metrics.get('recall', 'N/A'):.4f}\n"
            metrics_text += f"F1 Score: {self.metrics.get('f1_score', 'N/A'):.4f}\n"
            metrics_text += f"ROC AUC: {self.metrics.get('roc_auc', 'N/A'):.4f}\n"
            if 'confusion_matrix' in self.metrics:
                cm = self.metrics['confusion_matrix']
                metrics_text += f"\nConfusion Matrix:\n"
                metrics_text += f"TN: {cm[0][0]}, FP: {cm[0][1]}\n"
                metrics_text += f"FN: {cm[1][0]}, TP: {cm[1][1]}\n"
            self.metrics_text.setText(metrics_text)

        # Update features table
        self.features_table.setRowCount(0)
        if 'feature_importances' in self.metrics:
            importances = self.metrics['feature_importances']
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            self.features_table.setRowCount(len(sorted_features))
            for i, (feature, importance) in enumerate(sorted_features):
                self.features_table.setItem(i, 0, QTableWidgetItem(feature))
                self.features_table.setItem(i, 1, QTableWidgetItem(f"{importance:.6f}"))
                self.features_table.setItem(i, 2, QTableWidgetItem(self.feature_descriptions.get(feature, "N/A")))

    def update_dashboard(self):
        """Update the dashboard with current model metrics and visualizations"""
        if not self.model_loaded or not self.metrics:
            return

        # Update metric labels
        self.accuracy_label.setText(f"Accuracy: {self.metrics.get('accuracy', 'N/A'):.4f}")
        self.precision_label.setText(f"Precision: {self.metrics.get('precision', 'N/A'):.4f}")
        self.recall_label.setText(f"Recall: {self.metrics.get('recall', 'N/A'):.4f}")
        self.f1_label.setText(f"F1 Score: {self.metrics.get('f1_score', 'N/A'):.4f}")
        self.auc_label.setText(f"ROC AUC: {self.metrics.get('roc_auc', 'N/A'):.4f}")
        self.training_time_label.setText(f"Training Time: {self.agent.training_stats['training_time']:.2f}s")

        # Update confusion matrix plot
        self.confusion_matrix_canvas.clear()
        if 'confusion_matrix' in self.metrics:
            cm = np.array(self.metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=self.confusion_matrix_canvas.axes,
                        cbar_kws={'label': 'Count'},
                        annot_kws={"size": 12})
            self.confusion_matrix_canvas.axes.set_xlabel('Predicted Label', fontsize=10)
            self.confusion_matrix_canvas.axes.set_ylabel('True Label', fontsize=10)
            self.confusion_matrix_canvas.axes.set_title('Confusion Matrix', fontsize=12, pad=10)
        else:
            self.confusion_matrix_canvas.axes.text(0.5, 0.5, 'No Confusion Matrix Available',
                                                   ha='center', va='center', fontsize=12)
        self.confusion_matrix_canvas.draw()

        # Update ROC curve plot
        self.roc_curve_canvas.clear()
        if 'fpr' in self.metrics and 'tpr' in self.metrics:
            fpr, tpr = self.metrics['fpr'], self.metrics['tpr']
            roc_auc = auc(fpr, tpr)
            self.roc_curve_canvas.axes.plot(fpr, tpr, color=self.PRIMARY_COLOR, lw=2,
                                            label=f'ROC curve (AUC = {roc_auc:.2f})')
            self.roc_curve_canvas.axes.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.roc_curve_canvas.axes.set_xlim([0.0, 1.0])
            self.roc_curve_canvas.axes.set_ylim([0.0, 1.05])
            self.roc_curve_canvas.axes.set_xlabel('False Positive Rate')
            self.roc_curve_canvas.axes.set_ylabel('True Positive Rate')
            self.roc_curve_canvas.axes.set_title('ROC Curve')
            self.roc_curve_canvas.axes.legend(loc="lower right")
        else:
            self.roc_curve_canvas.axes.text(0.5, 0.5, 'No ROC Curve Available',
                                            ha='center', va='center', fontsize=12)
        self.roc_curve_canvas.draw()

        # Update feature importance plot
        self.feature_importance_canvas.clear()
        if 'feature_importances' in self.metrics:
            importances = self.metrics['feature_importances']
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            features = [item[0] for item in sorted_features]
            values = [item[1] for item in sorted_features]
            y_pos = np.arange(len(features))

            colors = [self.get_gradient_color(i/len(features), self.PRIMARY_COLOR, self.ACCENT_COLOR) for i in range(len(features))]

            bars = self.feature_importance_canvas.axes.barh(y_pos, values, align='center', color=colors)
            self.feature_importance_canvas.axes.set_yticks(y_pos)
            self.feature_importance_canvas.axes.set_yticklabels(features)
            self.feature_importance_canvas.axes.set_xlabel('Importance', fontsize=10)
            self.feature_importance_canvas.axes.set_title('Top 10 Feature Importance', fontsize=12, pad=10)

            for bar in bars:
                width = bar.get_width()
                self.feature_importance_canvas.axes.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                                         f'{width:.3f}',
                                                         ha='left', va='center', fontsize=9)
        else:
            self.feature_importance_canvas.axes.text(0.5, 0.5, 'No Feature Importance Available',
                                                     ha='center', va='center', fontsize=12)
        self.feature_importance_canvas.draw()

        # Update Q-value distribution plot
        self.q_value_canvas.clear()
        q_values = self.agent.q_table.flatten()
        sns.histplot(q_values, bins=30, kde=True, ax=self.q_value_canvas.axes)
        self.q_value_canvas.axes.set_title('Q-Value Distribution', fontsize=12, pad=10)
        self.q_value_canvas.axes.set_xlabel('Q-Value', fontsize=10)
        self.q_value_canvas.axes.set_ylabel('Frequency', fontsize=10)
        self.q_value_canvas.draw()

    def get_gradient_color(self, ratio, start_color, end_color):
        """Get a color from a gradient between two colors based on a ratio."""
        start_r, start_g, start_b = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
        end_r, end_g, end_b = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)

        r = int(start_r + (end_r - start_r) * ratio)
        g = int(start_g + (end_g - start_g) * ratio)
        b = int(start_b + (end_b - start_b) * ratio)

        return f'#{r:02x}{g:02x}{b:02x}'

    def prepare_input_features(self):
        task_size = self.task_size_input.value()
        cpu_demand = self.cpu_demand_input.value()
        memory_demand = self.memory_demand_input.value()
        network_latency = self.network_latency_input.value()
        io_operations = self.io_operations_input.value()
        disk_usage = self.disk_usage_input.value()
        num_connections = self.num_connections_input.value()
        priority_level = int(self.priority_level_input.currentText().split()[0])
        features = np.array([
            task_size, cpu_demand, memory_demand, network_latency,
            io_operations, disk_usage, num_connections, priority_level
        ]).reshape(1, -1)
        return features

    def plot_feature_vs_prediction(self):
        """Plot the selected feature against the predictions."""
        if self.data is None or 'Prediction' not in self.data.columns:
            QMessageBox.warning(self, "Warning", "No batch prediction results to plot.")
            return

        try:
            selected_feature = self.feature_combo.currentText()
            if selected_feature not in self.data.columns:
                QMessageBox.warning(self, "Warning", f"Feature '{selected_feature}' not found in data.")
                return

            self.feature_canvas.clear()
            sns.scatterplot(x=self.data[selected_feature], y=self.data['Prediction'], ax=self.feature_canvas.axes)
            self.feature_canvas.axes.set_title(f'{selected_feature} vs Prediction')
            self.feature_canvas.axes.set_xlabel(selected_feature)
            self.feature_canvas.axes.set_ylabel('Prediction')
            self.feature_canvas.draw()
            self.statusBar().showMessage(f"Plotted {selected_feature} vs Prediction.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot feature vs prediction: {str(e)}")

    def start_training(self):
        """Start the training process for the Q-Learning model."""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No training data loaded.")
            return

        try:
            self.start_time = time.time()
            learning_rate = self.learning_rate_input.value()
            discount_factor = self.discount_factor_input.value()
            epsilon = self.epsilon_input.value()
            episodes = self.episodes_input.value()

            self.train_log.append(f"Starting training with parameters:\n"
                                  f"Learning Rate: {learning_rate}\n"
                                  f"Discount Factor: {discount_factor}\n"
                                  f"Epsilon: {epsilon}\n"
                                  f"Episodes: {episodes}\n")

            # Initialize new agent with dynamic state space
            state_bins = {
                'task_size': [1, 25, 50, 75, 100],
                'cpu_demand': [0.1, 25, 50, 75, 100],
                'memory_demand': [1, 16, 32, 48, 64],
                'network_latency': [0.1, 50, 100, 150, 200],
                'io_operations': [1, 250, 500, 750, 1000],
                'disk_usage': [1, 25, 50, 75, 100],
                'num_connections': [1, 250, 500, 750, 1000],
                'priority_level': [0, 1]
            }
            self.agent = QLearningAgent(state_bins)
            version_name = self.model_version_input.text() or f"v{self.agent.version}"
            self.model_versions[version_name] = self.agent
            self.versions_combo.addItem(version_name)

            def training_task():
                rewards = []
                for episode in range(episodes):
                    # Reset environment (simplified, replace with actual environment)
                    features = self.current_data[self.FEATURE_NAMES].sample(1).values
                    state = self.agent.discretize_state(features)
                    done = False
                    episode_reward = 0

                    while not done:
                        action = self.agent.get_action(state, epsilon)
                        # Simulate next state and reward (replace with actual environment)
                        next_features = features + np.random.normal(0, 0.1, features.shape)
                        next_state = self.agent.discretize_state(next_features)
                        reward = self.calculate_reward(features, action)
                        episode_reward += reward

                        # Update Q-table
                        self.agent.update_q_table(state, action, reward, next_state, learning_rate, discount_factor)

                        state = next_state
                        features = next_features
                        done = np.random.random() < 0.1  # Simulate termination

                    rewards.append(episode_reward)
                    self.progress.emit(int((episode + 1) / episodes * 100))
                    self.message.emit(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward:.4f}")
                    QApplication.processEvents()

                return rewards

            self.worker = WorkerThread(training_task)
            self.worker.progress.connect(self.train_progress.setValue)
            self.worker.message.connect(self.train_log.append)
            self.worker.finished.connect(self.on_training_finished)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")

    def on_training_finished(self, rewards):
        """Handle training completion."""
        if rewards is None:
            QMessageBox.critical(self, "Error", "Training failed.")
            return

        training_time = time.time() - self.start_time
        self.agent.training_stats.update({
            'training_time': training_time,
            'episodes': len(rewards),
            'convergence_rate': np.mean(rewards[-100:]) if rewards else 0.0
        })

        stats_text = f"Training Time: {training_time:.2f}s\n"
        stats_text += f"Total Episodes: {len(rewards)}\n"
        stats_text += f"Average Reward (last 100): {self.agent.training_stats['convergence_rate']:.4f}\n"
        self.train_stats.setText(stats_text)

        self.train_log.append("Training completed successfully.")
        self.statusBar().showMessage("Training completed successfully.")
        self.save_model_btn.setEnabled(True)
        self.model_loaded = True
        self.predict_btn.setEnabled(True)
        self.batch_predict_btn.setEnabled(True)
        self.validate_model_btn.setEnabled(True)

        self.update_model_info()
        self.update_dashboard()

    def calculate_reward(self, features, action):
        """Calculate reward based on features and action (implement actual logic)"""
        # Example: Reward based on resource utilization and action
        cpu = features[0][1]
        memory = features[0][2]
        # Higher reward for better load balancing
        if action == 0:  # Server 1
            return -cpu / 100 - memory / 64
        else:  # Server 2
            return -cpu / 120 - memory / 80  # Different server capacities

    def load_training_data(self):
        """Load training data for the Q-Learning model."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Training Data", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.current_data = pd.read_csv(file_path)
                self.statusBar().showMessage(f"Training data loaded successfully from {file_path}")
                self.train_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load training data: {str(e)}")

    def save_model(self):
        """Save the trained Q-Learning model to a directory."""
        if self.agent is None:
            QMessageBox.warning(self, "Warning", "No model to save.")
            return

        try:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Model")
            if directory:
                model_path = os.path.join(directory, f'q_learning_agent_{self.agent.version}.pkl')
                joblib.dump(self.agent.q_table, model_path)

                # Save training stats
                stats_path = os.path.join(directory, f'training_stats_{self.agent.version}.json')
                with open(stats_path, 'w') as f:
                    json.dump(self.agent.training_stats, f)

                self.statusBar().showMessage(f"Model {self.agent.version} saved successfully to {directory}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def export_visualizations(self):
        """Export all visualizations at once"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Visualizations")
        if directory:
            try:
                if 'confusion_matrix' in self.metrics:
                    self.confusion_matrix_canvas.fig.savefig(
                        os.path.join(directory, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
                if 'fpr' in self.metrics:
                    self.roc_curve_canvas.fig.savefig(
                        os.path.join(directory, "roc_curve.png"), dpi=300, bbox_inches='tight')
                if 'feature_importances' in self.metrics:
                    self.feature_importance_canvas.fig.savefig(
                        os.path.join(directory, "feature_importance.png"), dpi=300, bbox_inches='tight')
                self.q_value_canvas.fig.savefig(
                    os.path.join(directory, "q_value_distribution.png"), dpi=300, bbox_inches='tight')
                self.performance_canvas.fig.savefig(
                    os.path.join(directory, "performance_trends.png"), dpi=300, bbox_inches='tight')
                self.statusBar().showMessage(f"All visualizations saved to {directory}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export visualizations: {str(e)}")

    def show_about(self):
        """Show information about the application."""
        about_text = """
        <h2>Load Balancer Decision System</h2>
        <p>Version 2.0</p>
        <p>Developed by Your Name or Organization</p>
        <p>Enhanced version with real-time monitoring, model versioning, and advanced visualization capabilities.</p>
        <p>For more information, visit our website or contact support.</p>
        """
        QMessageBox.about(self, "About", about_text)

    def show_documentation(self):
        """Show the documentation for the application."""
        doc_text = """
        <h2>Load Balancer Decision System Documentation</h2>
        <h3>Overview</h3>
        <p>
        This application provides a user interface for making load balancer decisions using a Q-Learning model.
        It supports real-time prediction, batch processing, model training, and performance monitoring.
        </p>

        <h3>Features</h3>
        <ul>
            <li><b>Dashboard:</b> Visualize model performance, confusion matrix, ROC curve, and feature importance.</li>
            <li><b>Make Predictions:</b> Input task parameters and get load balancer decisions.</li>
            <li><b>Batch Predictions:</b> Load CSV files and run predictions on multiple tasks at once.</li>
            <li><b>Model Training:</b> Train new Q-Learning models with custom parameters.</li>
            <li><b>Algorithm Comparison:</b> Compare Q-Learning with other load balancing algorithms.</li>
            <li><b>Real-Time Monitoring:</b> Track prediction rate, confidence, and system load.</li>
            <li><b>Model Versioning:</b> Manage and switch between different model versions.</li>
        </ul>

        <h3>How to Use</h3>
        <ol>
            <li>Load a trained model using the File menu or the Load Model button.</li>
            <li>Use the Prediction tab to input task parameters and get decisions.</li>
            <li>Use the Data tab to load CSV files for batch predictions.</li>
            <li>Use the Training tab to train new models with custom parameters.</li>
            <li>Use the Algorithm tab to compare different load balancing algorithms.</li>
            <li>Use the Monitoring tab to track real-time metrics and switch model versions.</li>
        </ol>

        <h3>Input Parameters</h3>
        <ul>
            <li><b>Task Size:</b> Size/complexity of the task (1-100).</li>
            <li><b>CPU Demand:</b> Percentage of CPU resources required (0.1-100.0).</li>
            <li><b>Memory Demand:</b> Amount of memory required in GB (1.0-64.0).</li>
            <li><b>Network Latency:</b> Network delay in milliseconds (0.1-200.0).</li>
            <li><b>I/O Operations:</b> Input/output operations per second (1.0-1000.0).</li>
            <li><b>Disk Usage:</b> Percentage of disk space used (1.0-100.0).</li>
            <li><b>Number of Connections:</b> Number of active network connections (1-1000).</li>
            <li><b>Priority Level:</b> Task priority level (0 for Low, 1 for High).</li>
        </ul>

        <h3>Visualizations</h3>
        <p>
        The application provides several visualizations to help understand model performance:
        </p>
        <ul>
            <li>Confusion Matrix</li>
            <li>ROC Curve</li>
            <li>Feature Importance</li>
            <li>Q-Value Distribution</li>
            <li>Prediction Confidence Trends</li>
        </ul>

        <h3>Exporting Results</h3>
        <p>
        You can export prediction history, batch results, and visualizations using the File menu or the Export buttons.
        </p>

        <h3>Support</h3>
        <p>
        For support, please contact support@example.com or visit our website.
        </p>
        """
        QMessageBox.about(self, "Documentation", doc_text)

    def on_load_model_clicked(self):
        """Handle the Load Model button click."""
        directory = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if directory:
            self.load_model(directory)

    def on_load_data_clicked(self):
        """Handle the Load Data button click."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.update_data_table()
                self.batch_predict_btn.setEnabled(True)
                self.statusBar().showMessage(f"Data loaded successfully from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def update_data_table(self):
        """Update the data table with loaded data."""
        if self.data is None:
            return
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(len(self.data.columns))
        self.data_table.setHorizontalHeaderLabels(self.data.columns)
        for i, row in self.data.iterrows():
            self.data_table.insertRow(i)
            for j, val in enumerate(row):
                self.data_table.setItem(i, j, QTableWidgetItem(str(val)))

    def on_batch_predict_clicked(self):
        """Run batch predictions in a background thread."""
        if self.data is None or not self.model_loaded:
            QMessageBox.warning(self, "Warning", "No data loaded or model not ready.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        def batch_predict_task():
            try:
                features = self.data[self.FEATURE_NAMES].values
                predictions = []
                for i, f in enumerate(features):
                    prediction, _ = self.make_prediction(f.reshape(1, -1))
                    predictions.append(prediction)
                    self.progress.emit(int((i + 1) / len(features) * 100))
                return predictions
            except Exception as e:
                logging.error(f"Batch prediction error: {e}")
                self.message.emit(str(e))
                return None

        self.worker = WorkerThread(batch_predict_task)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.message.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.on_batch_predict_finished)
        self.worker.start()

    def on_batch_predict_finished(self, predictions):
        """Handle the result of batch prediction."""
        self.progress_bar.setVisible(False)
        if predictions is None:
            QMessageBox.critical(self, "Error", "Failed to run batch predictions.")
            return

        self.data['Prediction'] = predictions
        self.batch_results = self.data.copy()
        self.export_data_btn.setEnabled(True)
        self.update_data_table()
        self.update_distribution_plot()
        self.statusBar().showMessage("Batch prediction completed.")

    def update_distribution_plot(self):
        """Update the distribution plot with batch prediction results."""
        if self.data is None or 'Prediction' not in self.data.columns:
            return
        self.dist_canvas.clear()
        sns.countplot(x='Prediction', data=self.data, ax=self.dist_canvas.axes)
        self.dist_canvas.axes.set_title('Prediction Distribution')
        self.dist_canvas.draw()

    def on_predict_clicked(self):
        """Handle the Make Prediction button click."""
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "No model loaded.")
            return

        try:
            features = self.prepare_input_features()
            prediction, probability = self.make_prediction(features)

            # Update UI with prediction result
            self.prediction_result_label.setText(f"Prediction: {'Server 1' if prediction == 0 else 'Server 2'}")
            self.prediction_prob_label.setText(f"Confidence: {probability:.4f}")
            self.probability_bar.setValue(int(probability * 100))

            # Add to history
            entry = {
                'time': datetime.now().isoformat(),
                'prediction': int(prediction),
                'probability': float(probability),
                'features': features.tolist()[0],
                'state': self.agent.discretize_state(features),
                'model_version': self.agent.version
            }
            self.prediction_history.append(entry)
            self.update_history_table()

            # Update details
            details = f"Task Size: {features[0][0]}\n"
            details += f"CPU Demand: {features[0][1]}%\n"
            details += f"Memory Demand: {features[0][2]}GB\n"
            details += f"Network Latency: {features[0][3]}ms\n"
            details += f"I/O Operations: {features[0][4]}/s\n"
            details += f"Disk Usage: {features[0][5]}%\n"
            details += f"Connections: {features[0][6]}\n"
            details += f"Priority: {features[0][7]}\n"
            details += f"State: {entry['state']}\n"
            details += f"Model Version: {entry['model_version']}"
            self.prediction_details.setText(details)

            self.statusBar().showMessage("Prediction completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make prediction: {str(e)}")

    def make_prediction(self, features):
        """Make a prediction using the Q-Learning agent."""
        try:
            state = self.agent.discretize_state(features)
            action = self.agent.get_action(state, epsilon=0.0)
            # Calculate probability from Q-values
            q_values = self.agent.q_table[state]
            exp_q = np.exp(q_values - np.max(q_values))  # Stabilized softmax
            probability = exp_q[action] / np.sum(exp_q)
            return action, probability
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return 0, 0.5  # Default fallback

    def update_history_table(self):
        """Update the prediction history table."""
        self.history_table.setRowCount(0)
        for entry in self.prediction_history[-10:]:  # Show last 10 entries
            row_position = self.history_table.rowCount()
            self.history_table.insertRow(row_position)
            self.history_table.setItem(row_position, 0, QTableWidgetItem(entry['time']))
            self.history_table.setItem(row_position, 1, QTableWidgetItem(str(entry['prediction'])))
            self.history_table.setItem(row_position, 2, QTableWidgetItem(f"{entry['probability']:.4f}"))
            features_text = f"Size: {entry['features'][0]}, CPU: {entry['features'][1]}%"
            self.history_table.setItem(row_position, 3, QTableWidgetItem(features_text))
            self.history_table.setItem(row_position, 4, QTableWidgetItem(str(entry['state'])))
            self.history_table.setItem(row_position, 5, QTableWidgetItem(entry['model_version']))

    def on_reset_inputs_clicked(self):
        """Reset all input fields to default values."""
        self.task_size_input.setValue(10)
        self.cpu_demand_input.setValue(20.0)
        self.memory_demand_input.setValue(16.0)
        self.network_latency_input.setValue(50.0)
        self.io_operations_input.setValue(100.0)
        self.disk_usage_input.setValue(30.0)
        self.num_connections_input.setValue(50)
        self.priority_level_input.setCurrentIndex(0)
        self.statusBar().showMessage("Inputs reset to default values.")

    def save_prediction_history(self):
        """Save the prediction history to a JSON file."""
        if not self.prediction_history:
            QMessageBox.warning(self, "Warning", "No prediction history to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Prediction History", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.prediction_history, f, indent=4)
                self.statusBar().showMessage(f"Prediction history saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save prediction history: {str(e)}")

    def save_current_prediction(self):
        """Save the current prediction to a file."""
        if not hasattr(self, 'current_prediction'):
            QMessageBox.warning(self, "Warning", "No current prediction to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Current Prediction", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_prediction, f, indent=4)
                self.statusBar().showMessage(f"Current prediction saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save current prediction: {str(e)}")

    def validate_input(self):
        """Validate input fields and enable/disable the predict button."""
        self.predict_btn.setEnabled(self.model_loaded)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoadBalancerUI()
    window.show()
    sys.exit(app.exec_())



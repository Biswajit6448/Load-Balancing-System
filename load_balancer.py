

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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QComboBox, QTabWidget,
                             QGridLayout, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
                             QTextEdit, QSpinBox, QDoubleSpinBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QProgressBar, QSplitter)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from datetime import datetime
import logging
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

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
            self.finished.emit(e)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network model"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """DQN Agent for load balancing"""
    def __init__(self, state_dim=10, action_dim=2, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, target_update=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def get_q_values(self, state):
        """Get Q-values for probability estimation"""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return F.softmax(q_values, dim=1).max().item()

    def update(self):
        """Train the network"""
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class LoadBalancerUI(QMainWindow):
    """Enhanced Load Balancer Prediction System with DQN"""
    FEATURE_NAMES = [
        'task_size', 'cpu_demand', 'memory_demand', 'network_latency',
        'io_operations', 'disk_usage', 'num_connections', 'priority_level',
        'current_server_cpu', 'queue_length'  # New features
    ]

    # Expected feature ranges for scaling
    FEATURE_RANGES = {
        'task_size': (1, 100),
        'cpu_demand': (0.1, 100),
        'memory_demand': (1, 64),
        'network_latency': (0.1, 200),
        'io_operations': (1, 1000),
        'disk_usage': (1, 100),
        'num_connections': (1, 1000),
        'priority_level': (0, 1),
        'current_server_cpu': (0, 100),
        'queue_length': (0, 1000)
    }

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
        self.data = None  # Initialize the data attribute
        self.scaler = StandardScaler()  # For scaling features
        self.file_path = None  # ✅ Store the loaded file path

        # Setup UI
        self.init_ui()
        self.init_styles()

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

        # Status bar
        self.statusBar().setStyleSheet("background-color: #f0f0f0; color: #333;")
        self.statusBar().showMessage("Ready")

        # Add menu bar
        self.create_menu_bar()

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
            QSpinBox, QDoubleSpinBox, QComboBox {{
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
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

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
        model_status_layout.addWidget(self.model_status_label)
        model_status_layout.addWidget(self.load_model_btn)

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

        # ROC curve
        self.roc_curve_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        roc_curve_group = QGroupBox("ROC Curve")
        roc_curve_layout = QVBoxLayout(roc_curve_group)
        roc_curve_layout.addWidget(self.roc_curve_canvas)

        # Feature importance
        self.feature_importance_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        feature_importance_group = QGroupBox("Feature Importance")
        feature_importance_layout = QVBoxLayout(feature_importance_group)
        feature_importance_layout.addWidget(self.feature_importance_canvas)

        # Q-Value visualization
        self.q_value_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        q_value_group = QGroupBox("Q-Value Distribution")
        q_value_layout = QVBoxLayout(q_value_group)
        q_value_layout.addWidget(self.q_value_canvas)

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

        # New features
        self.current_server_cpu_input = QDoubleSpinBox()
        self.current_server_cpu_input.setRange(0.0, 100.0)
        self.current_server_cpu_input.setValue(50.0)
        self.current_server_cpu_input.setSingleStep(0.1)
        self.current_server_cpu_input.setToolTip("Current server CPU usage (%)")
        self.current_server_cpu_input.valueChanged.connect(self.validate_input)

        self.queue_length_input = QSpinBox()
        self.queue_length_input.setRange(0, 1000)
        self.queue_length_input.setValue(0)
        self.queue_length_input.setToolTip("Current queue length")
        self.queue_length_input.valueChanged.connect(self.validate_input)

        # Add rows to form
        input_layout.addRow("Task Size:", self.task_size_input)
        input_layout.addRow("CPU Demand (%):", self.cpu_demand_input)
        input_layout.addRow("Memory Demand (GB):", self.memory_demand_input)
        input_layout.addRow("Network Latency (ms):", self.network_latency_input)
        input_layout.addRow("I/O Operations (ops/s):", self.io_operations_input)
        input_layout.addRow("Disk Usage (%):", self.disk_usage_input)
        input_layout.addRow("Number of Connections:", self.num_connections_input)
        input_layout.addRow("Priority Level:", self.priority_level_input)
        input_layout.addRow("Current Server CPU (%):", self.current_server_cpu_input)
        input_layout.addRow("Queue Length:", self.queue_length_input)

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
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Time", "Prediction", "Confidence", "Key Features", "State"])
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

    def adjust_color(self, hex_color, amount):
        """Adjust the brightness of a hex color by amount (-255 to 255)"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = [min(255, max(0, x + amount)) for x in rgb]
        return '{:02x}{:02x}{:02x}'.format(*new_rgb)

    def create_model_info_tab(self):
        """Create the model information tab"""
        model_info_tab = QWidget()
        model_info_layout = QVBoxLayout(model_info_tab)

        # Model details section
        model_details_group = QGroupBox("Model Configuration")
        model_details_layout = QFormLayout(model_details_group)
        self.model_type_label = QLabel("Type: DQN")
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
            'priority_level': "Priority level of the task (0-1)",
            'current_server_cpu': "Current CPU usage of the server",
            'queue_length': "Number of tasks in queue"
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

        data_load_layout.addWidget(self.load_data_btn)
        data_load_layout.addWidget(self.batch_predict_btn)

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
        dist_layout.addWidget(self.dist_canvas)
        self.results_tabs.addTab(dist_tab, "Distribution")

        # Feature vs Prediction tab
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(self.FEATURE_NAMES)
        self.feature_plot_btn = QPushButton("Plot Feature vs Prediction")
        self.feature_plot_btn.clicked.connect(self.plot_feature_vs_prediction)
        self.feature_canvas = MplCanvas(self, width=10, height=4, dpi=100)
        feature_layout.addWidget(self.feature_combo)
        feature_layout.addWidget(self.feature_plot_btn)
        feature_layout.addWidget(self.feature_canvas)
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
        self.algorithm_combo.addItems(["DQN", "Round Robin", "Least Connections", "Weighted Round Robin", "Random"])

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
            "DQN": self.metrics if self.metrics else None,
            "Round Robin": None,
            "Least Connections": None,
            "Weighted Round Robin": None,
            "Random": None
        }

        # Update the table
        self.update_algorithm_table()

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
        if selected_algorithm == "DQN":
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            for i in range(1, 101):
                QTimer.singleShot(i * 20, lambda i=i: self.progress_bar.setValue(i))
                QApplication.processEvents()

            # Simulate random metrics
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

        best_algorithm = max(valid_algorithms, key=lambda alg: self.algorithm_performance[alg]['f1_score'])
        self.best_algorithm_label.setText(f"Best: {best_algorithm} (F1: {self.algorithm_performance[best_algorithm]['f1_score']:.4f})")
        self.select_best_btn.setEnabled(True)

        for row in range(self.algorithm_table.rowCount()):
            if self.algorithm_table.item(row, 0).text() == best_algorithm:
                for col in range(self.algorithm_table.columnCount()):
                    self.algorithm_table.item(row, col).setBackground(QColor(173, 216, 230))
            else:
                for col in range(self.algorithm_table.columnCount()):
                    self.algorithm_table.item(row, col).setBackground(QColor(255, 255, 255))

    def select_best_algorithm(self):
        """Select the best algorithm as the active one"""
        best_algorithm = self.best_algorithm_label.text().split(":")[1].split("(")[0].strip()
        if best_algorithm == "DQN":
            QMessageBox.information(self, "Info", "DQN is already the selected algorithm")
            return

        reply = QMessageBox.question(
            self, "Confirm Selection",
            f"Are you sure you want to switch to {best_algorithm}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self, "Algorithm Changed",
                f"Active algorithm changed to {best_algorithm}\n"
                "Note: This is a demonstration. In a real implementation, you would "
                "need to implement the other algorithms and properly switch between them."
            )

    def create_training_tab(self):
        """Create the training tab for model retraining"""
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 0.1)
        self.learning_rate_input.setValue(0.001)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setToolTip("Learning rate for DQN")

        self.discount_factor_input = QDoubleSpinBox()
        self.discount_factor_input.setRange(0.1, 0.99)
        self.discount_factor_input.setValue(0.99)
        self.discount_factor_input.setSingleStep(0.01)
        self.discount_factor_input.setToolTip("Discount factor for future rewards")

        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(0.01, 1.0)
        self.epsilon_input.setValue(1.0)
        self.epsilon_input.setSingleStep(0.01)
        self.epsilon_input.setToolTip("Initial exploration rate (epsilon)")

        self.epsilon_decay_input = QDoubleSpinBox()
        self.epsilon_decay_input.setRange(0.9, 0.999)
        self.epsilon_decay_input.setValue(0.995)
        self.epsilon_decay_input.setSingleStep(0.001)
        self.epsilon_decay_input.setToolTip("Epsilon decay rate")

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(10, 10000)
        self.episodes_input.setValue(1000)
        self.episodes_input.setToolTip("Number of training episodes")

        params_layout.addRow("Learning Rate:", self.learning_rate_input)
        params_layout.addRow("Discount Factor:", self.discount_factor_input)
        params_layout.addRow("Initial Epsilon:", self.epsilon_input)
        params_layout.addRow("Epsilon Decay:", self.epsilon_decay_input)
        params_layout.addRow("Training Episodes:", self.episodes_input)

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

        # Training progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("background-color: white;")
        progress_layout.addWidget(self.train_progress)
        progress_layout.addWidget(self.train_log)

        training_layout.addWidget(params_group)
        training_layout.addWidget(controls_group)
        training_layout.addWidget(progress_group, 1)
        self.tabs.addTab(training_tab, QIcon("icon_training.png"), "Model Training")

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
        """Load DQN model from directory"""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            model_path = os.path.join(model_dir, 'dqn_agent.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.agent = DQNAgent(state_dim=len(self.FEATURE_NAMES), action_dim=2)
            self.agent.load(model_path)
            self.progress_bar.setValue(50)
            QApplication.processEvents()

            metrics_path = os.path.join(model_dir, 'model_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)

            self.progress_bar.setValue(70)
            QApplication.processEvents()

            history_path = os.path.join(model_dir, 'prediction_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.prediction_history = json.load(f)
                    self.update_history_table()

            self.progress_bar.setValue(90)
            QApplication.processEvents()

            self.model_loaded = True
            self.predict_btn.setEnabled(True)
            self.batch_predict_btn.setEnabled(True)
            self.model_status_label.setText("Model loaded: DQN")
            self.model_status_label.setStyleSheet("color: blue;")
            self.update_model_info()
            self.update_dashboard()
            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            self.statusBar().showMessage(f"Model loaded successfully from {model_dir}", 5000)
            return True

        except FileNotFoundError as e:
            self.statusBar().showMessage(f"File not found: {str(e)}", 5000)
            QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {str(e)}", 5000)
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
        return False

    def update_model_info(self):
        """Update the model information tab with current model details"""
        if not self.model_loaded:
            return

        params_text = f"Learning Rate: {self.agent.optimizer.param_groups[0]['lr']}\n"
        params_text += f"Discount Factor: {self.agent.gamma}\n"
        params_text += f"Exploration Rate: {self.agent.epsilon}\n"
        params_text += f"State Dim: {self.agent.state_dim}\n"
        params_text += f"Action Dim: {self.agent.action_dim}"
        self.model_params_text.setText(params_text)

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

        self.features_table.setRowCount(0)
        if 'feature_importances' in self.metrics:
            importances = self.metrics['feature_importances']
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            self.features_table.setRowCount(len(sorted_features))
            for i, (feature, importance) in enumerate(sorted_features):
                self.features_table.setItem(i, 0, QTableWidgetItem(feature))
                self.features_table.setItem(i, 1, QTableWidgetItem(f"{importance:.6f}"))
                self.features_table.setItem(i, 2, QTableWidgetItem(self.feature_descriptions.get(feature, "")))

    def update_dashboard(self):
        """Update the dashboard with current model metrics and visualizations"""
        if not self.model_loaded or not self.metrics:
            return

        self.accuracy_label.setText(f"Accuracy: {self.metrics.get('accuracy', 'N/A'):.4f}")
        self.precision_label.setText(f"Precision: {self.metrics.get('precision', 'N/A'):.4f}")
        self.recall_label.setText(f"Recall: {self.metrics.get('recall', 'N/A'):.4f}")
        self.f1_label.setText(f"F1 Score: {self.metrics.get('f1_score', 'N/A'):.4f}")
        self.auc_label.setText(f"ROC AUC: {self.metrics.get('roc_auc', 'N/A'):.4f}")

        if 'confusion_matrix' in self.metrics:
            self.confusion_matrix_canvas.axes.clear()
            cm = np.array(self.metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.confusion_matrix_canvas.axes,
                        cbar_kws={'label': 'Count'}, annot_kws={"size": 12})
            self.confusion_matrix_canvas.axes.set_xlabel('Predicted Label', fontsize=10)
            self.confusion_matrix_canvas.axes.set_ylabel('True Label', fontsize=10)
            self.confusion_matrix_canvas.axes.set_title('Confusion Matrix', fontsize=12, pad=10)
            self.confusion_matrix_canvas.draw()

        if 'feature_importances' in self.metrics:
            self.feature_importance_canvas.axes.clear()
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
                                                         f'{width:.3f}', ha='left', va='center', fontsize=9)
            self.feature_importance_canvas.draw()

        self.q_value_canvas.axes.clear()
        sample_states = np.random.uniform(0, 100, (100, len(self.FEATURE_NAMES)))
        q_values = []
        for state in sample_states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q = self.agent.policy_net(state_tensor).flatten().numpy()
            q_values.extend(q)
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
        current_server_cpu = self.current_server_cpu_input.value()
        queue_length = self.queue_length_input.value()
        features = np.array([
            task_size, cpu_demand, memory_demand, network_latency,
            io_operations, disk_usage, num_connections, priority_level,
            current_server_cpu, queue_length
        ])
        return features

    def plot_feature_vs_prediction(self):
        if self.data is None or 'Prediction' not in self.data.columns:
            QMessageBox.warning(self, "Warning", "No batch prediction results to plot.")
            return
        try:
            selected_feature = self.feature_combo.currentText()
            if selected_feature not in self.data.columns:
                QMessageBox.warning(self, "Warning", f"Feature '{selected_feature}' not found in data.")
                return
            self.feature_canvas.axes.clear()
            sns.scatterplot(x=self.data[selected_feature], y=self.data['Prediction'], ax=self.feature_canvas.axes)
            self.feature_canvas.axes.set_title(f'{selected_feature} vs Prediction')
            self.feature_canvas.axes.set_xlabel(selected_feature)
            self.feature_canvas.axes.set_ylabel('Prediction')
            self.feature_canvas.draw()
            self.statusBar().showMessage(f"Plotted {selected_feature} vs Prediction.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot feature vs prediction: {str(e)}")

    def start_training(self):
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No training data loaded.")
            return
        try:
            lr = self.learning_rate_input.value()
            gamma = self.discount_factor_input.value()
            epsilon = self.epsilon_input.value()
            epsilon_decay = self.epsilon_decay_input.value()
            episodes = self.episodes_input.value()
            self.agent = DQNAgent(state_dim=len(self.FEATURE_NAMES), action_dim=2, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay)
            self.train_log.append(f"Starting training with parameters:\n"
                                  f"Learning Rate: {lr}\n"
                                  f"Discount Factor: {gamma}\n"
                                  f"Epsilon: {epsilon}\n"
                                  f"Epsilon Decay: {epsilon_decay}\n"
                                  f"Episodes: {episodes}\n")
            for episode in range(episodes):
                state = np.random.uniform(0, 100, len(self.FEATURE_NAMES))
                action = self.agent.get_action(state)
                next_state = state + np.random.normal(0, 1, len(self.FEATURE_NAMES))
                reward = -state[3] - state[1] + 10 * state[7]
                done = np.random.rand() < 0.1
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                self.agent.update()
                self.train_progress.setValue(int((episode + 1) / episodes * 100))
                QApplication.processEvents()
                if episode % 100 == 0:
                    self.train_log.append(f"Episode {episode}/{episodes} completed.")
            self.train_log.append("Training completed successfully.")
            self.statusBar().showMessage("Training completed successfully.")
            self.model_loaded = True
            self.save_model_btn.setEnabled(True)
            self.predict_btn.setEnabled(True)
            self.batch_predict_btn.setEnabled(True)
            self.model_status_label.setText("Model trained: DQN")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")

    def load_training_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Training Data", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.current_data = pd.read_csv(file_path)
                self.statusBar().showMessage(f"Training data loaded successfully from {file_path}")
                self.train_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load training data: {str(e)}")

    def save_model(self):
        if self.agent is None:
            QMessageBox.warning(self, "Warning", "No model to save.")
            return
        try:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Model")
            if directory:
                model_path = os.path.join(directory, 'dqn_agent.pth')
                self.agent.save(model_path)
                self.statusBar().showMessage(f"Model saved successfully to {directory}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def show_about(self):
        about_text = """
        <h2>Load Balancer Decision System</h2>
        <p>Version 1.0</p>
        <p>Developed by Your Name or Organization</p>
        <p>This application uses DQN to make intelligent load balancing decisions based on input parameters.</p>
        <p>For more information, visit our website or contact support.</p>
        """
        QMessageBox.about(self, "About", about_text)

    def show_documentation(self):
        documentation_url = "https://github.com/DebmalyaRay9989/Load_Balancer/blob/main/Load%20balancers.pdf"
        QDesktopServices.openUrl(QUrl(documentation_url))

    def make_prediction(self, features):
        if not self.model_loaded:
            return None, None
        try:
            action = self.agent.get_action(features)
            probability = self.agent.get_q_values(features)
            return action, probability
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None, None

    def update_history_table(self):
        self.history_table.setRowCount(0)
        for i, entry in enumerate(reversed(self.prediction_history[-10:])):
            row_position = self.history_table.rowCount()
            self.history_table.insertRow(row_position)
            self.history_table.setItem(row_position, 0, QTableWidgetItem(entry['time']))
            prediction_item = QTableWidgetItem("Load Balancer A" if entry['prediction'] == 0 else "Load Balancer B")
            prediction_item.setForeground(QColor("green" if entry['prediction'] == 0 else "red"))
            self.history_table.setItem(row_position, 1, prediction_item)
            self.history_table.setItem(row_position, 2, QTableWidgetItem(f"{entry['probability']:.4f}"))
            self.history_table.setItem(row_position, 3, QTableWidgetItem(entry['key_features']))

    def on_load_model_clicked(self):
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if model_dir:
            self.load_model(model_dir)

    def on_predict_clicked(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load or train a model first")
            return
        try:
            features = self.prepare_input_features()
            prediction, probability = self.make_prediction(features)
            prediction_text = "Load Balancer A" if prediction == 0 else "Load Balancer B"
            self.prediction_result_label.setText(prediction_text)
            self.prediction_result_label.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {'green' if prediction == 0 else 'red'}; padding: 20px;"
            )
            self.prediction_prob_label.setText(f"Probability: {probability:.4f}")
            self.probability_bar.setValue(int(probability * 100))
            details = "Input Parameters:\n"
            for name, value in zip(self.FEATURE_NAMES, features):
                details += f"{name}: {value}\n"
            self.prediction_details.setText(details)
            self.prediction_history.append({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': prediction,
                'probability': probability,
                'key_features': details
            })
            self.update_history_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make prediction: {str(e)}")

    def on_reset_inputs_clicked(self):
        self.task_size_input.setValue(10)
        self.cpu_demand_input.setValue(20.0)
        self.memory_demand_input.setValue(16.0)
        self.network_latency_input.setValue(50.0)
        self.io_operations_input.setValue(100.0)
        self.disk_usage_input.setValue(30.0)
        self.num_connections_input.setValue(50)
        self.priority_level_input.setCurrentIndex(0)
        self.current_server_cpu_input.setValue(50.0)
        self.queue_length_input.setValue(0)

    def on_load_data_clicked(self):
        """Load a CSV file for batch predictions, handle missing features, and scale data."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            self.statusBar().showMessage("No file selected", 5000)
            return
        try:
            self.data = pd.read_csv(file_path)
            self.file_path = file_path  # ✅ Save file path
            logging.info(f"Loaded CSV file: {file_path}, shape: {self.data.shape}")

            missing_features = [f for f in self.FEATURE_NAMES if f not in self.data.columns]
            if missing_features:
                logging.info(f"Adding missing features with default values: {missing_features}")
                for feature in missing_features:
                    if feature == 'current_server_cpu':
                        self.data[feature] = 50.0
                    elif feature == 'queue_length':
                        self.data[feature] = 0
                    else:
                        raise ValueError(f"Unexpected missing feature: {feature}")

            if 'priority_level' in self.data.columns and self.data['priority_level'].dtype != int:
                logging.info("Binarizing priority_level")
                self.data['priority_level'] = (self.data['priority_level'] > 0.5).astype(int)

            self.data = self.data[self.FEATURE_NAMES]

            features_to_scale = self.data[self.FEATURE_NAMES].values
            self.scaler.fit(features_to_scale)
            scaled_features = self.scaler.inverse_transform(features_to_scale)
            for i, feature in enumerate(self.FEATURE_NAMES):
                min_val, max_val = self.FEATURE_RANGES[feature]
                scaled_features[:, i] = scaled_features[:, i] * (max_val - min_val) + min_val
                if feature == 'priority_level':
                    scaled_features[:, i] = (scaled_features[:, i] > 0.5).astype(int)
            self.data[self.FEATURE_NAMES] = scaled_features

            self.data_table.setRowCount(self.data.shape[0])
            self.data_table.setColumnCount(len(self.FEATURE_NAMES))
            self.data_table.setHorizontalHeaderLabels(self.FEATURE_NAMES)
            for row in range(self.data.shape[0]):
                for col in range(len(self.FEATURE_NAMES)):
                    self.data_table.setItem(row, col, QTableWidgetItem(f"{self.data.iat[row, col]:.4f}"))
            self.data_table.resizeColumnsToContents()
            self.statusBar().showMessage(f"Data loaded successfully from {file_path} ({self.data.shape[0]} rows)", 5000)
            self.batch_predict_btn.setEnabled(self.model_loaded)
        except ValueError as ve:
            logging.error(f"Data validation error: {str(ve)}")
            QMessageBox.critical(self, "Error", f"Data validation error: {str(ve)}")
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def on_batch_predict_clicked(self):
        """Run batch predictions on loaded data."""
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load or train a model first")
            return
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        if not hasattr(self, 'file_path') or self.file_path is None:  # ✅ Safety check
            QMessageBox.warning(self, "Warning", "No data file loaded.")
            return

        try:
            required_features = self.FEATURE_NAMES
            missing_features = [f for f in required_features if f not in self.data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            features = self.data[required_features].values

            predictions = []
            probabilities = []

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(features))
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            chunk_size = 100
            for i in range(0, len(features), chunk_size):
                chunk = features[i:i + chunk_size]
                for f in chunk:
                    if f.shape[0] != len(self.FEATURE_NAMES):
                        logging.warning(f"Invalid feature vector size: {f.shape[0]}")
                        predictions.append(None)
                        probabilities.append(None)
                        continue
                    action, prob = self.make_prediction(f)
                    if action is None or prob is None:
                        logging.warning(f"Prediction failed for feature vector: {f}")
                        predictions.append(None)
                        probabilities.append(None)
                        continue
                    predictions.append(action)
                    probabilities.append(prob)
                self.progress_bar.setValue(min(i + chunk_size, len(features)))
                QApplication.processEvents()

            if len(predictions) != len(self.data):
                logging.warning(f"Mismatch in predictions length: {len(predictions)} vs {len(self.data)}")
                predictions.extend([None] * (len(self.data) - len(predictions)))
                probabilities.extend([None] * (len(self.data) - len(probabilities)))

            self.data['Prediction'] = predictions
            self.data['Probability'] = probabilities

            # ✅ Fixed: Use manual header label extraction
            current_header_labels = [
                self.data_table.horizontalHeaderItem(i).text() if self.data_table.horizontalHeaderItem(i) else ""
                for i in range(self.data_table.columnCount())
            ]
            if 'Prediction' not in current_header_labels:
                self.data_table.setColumnCount(len(self.FEATURE_NAMES) + 2)
                self.data_table.setHorizontalHeaderLabels(self.FEATURE_NAMES + ['Prediction', 'Probability'])

            for row in range(self.data.shape[0]):
                pred_item = QTableWidgetItem(str(self.data['Prediction'].iloc[row]) if pd.notna(self.data['Prediction'].iloc[row]) else "None")
                prob_item = QTableWidgetItem(f"{self.data['Probability'].iloc[row]:.4f}" if pd.notna(self.data['Probability'].iloc[row]) else "None")
                self.data_table.setItem(row, len(self.FEATURE_NAMES), pred_item)
                self.data_table.setItem(row, len(self.FEATURE_NAMES) + 1, prob_item)
            self.data_table.resizeColumnsToContents()

            # ✅ Fixed: Use self.file_path
            output_path = os.path.splitext(self.file_path)[0] + '_predictions.csv'
            self.data.to_csv(output_path, index=False)
            logging.info(f"Batch predictions saved to {output_path}")

            self.results_tabs.setCurrentIndex(0)
            self.dist_canvas.clear()
            valid_predictions = self.data['Prediction'].dropna()
            if not valid_predictions.empty:
                sns.histplot(valid_predictions, bins=2, kde=True, ax=self.dist_canvas.axes)
                self.dist_canvas.axes.set_title('Prediction Distribution', fontsize=12)
                self.dist_canvas.axes.set_xlabel('Prediction (0: Load Balancer A, 1: Load Balancer B)', fontsize=10)
                self.dist_canvas.axes.set_ylabel('Count', fontsize=10)
                self.dist_canvas.draw()
            else:
                logging.warning("No valid predictions for visualization")
                QMessageBox.warning(self, "Warning", "No valid predictions to visualize")

            self.statusBar().showMessage(f"Batch predictions completed ({len(valid_predictions)} valid predictions). Saved to {output_path}", 5000)
        except ValueError as ve:
            logging.error(f"Data validation error: {str(ve)}")
            QMessageBox.critical(self, "Error", f"Data validation error: {str(ve)}")
        except Exception as e:
            logging.error(f"Failed to make batch predictions: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to make batch predictions: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def save_prediction_history(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Prediction History", "", "JSON Files (*.json)")
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(self.prediction_history, f)
                self.statusBar().showMessage(f"Prediction history saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save prediction history: {str(e)}")

    def validate_input(self):
        inputs_valid = all([
            self.task_size_input.value() > 0,
            self.cpu_demand_input.value() > 0,
            self.memory_demand_input.value() > 0,
            self.network_latency_input.value() > 0,
            self.io_operations_input.value() > 0,
            self.disk_usage_input.value() > 0,
            self.num_connections_input.value() > 0,
            self.current_server_cpu_input.value() >= 0,
            self.queue_length_input.value() >= 0
        ])
        self.predict_btn.setEnabled(inputs_valid)

    def save_current_prediction(self):
        if self.prediction_result_label.text() == "No prediction yet":
            QMessageBox.warning(self, "Warning", "No prediction to save")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Current Prediction", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f"Prediction: {self.prediction_result_label.text()}\n")
                    f.write(f"Confidence: {self.prediction_prob_label.text()}\n")
                    f.write(f"Details:\n{self.prediction_details.toPlainText()}")
                self.statusBar().showMessage(f"Current prediction saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save current prediction: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    window = LoadBalancerUI()
    window.show()
    sys.exit(app.exec_())






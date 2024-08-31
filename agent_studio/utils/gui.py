import json
from typing import Callable

from PyQt6 import Qsci
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class JSONEditor(Qsci.QsciScintilla):
    def __init__(self, editable=True, parent=None):
        super().__init__(parent)
        self.setLexer(Qsci.QsciLexerJSON(self))
        self.setReadOnly(not editable)

    def setText(self, text: str | dict):
        if isinstance(text, dict):
            text = json.dumps(text, indent=4)
        super().setText(text)
        self.setReadOnly(True)

    def getText(self) -> str:
        return self.text()


class InputDialog(QDialog):
    def __init__(self, title: str, message: str, callback: Callable | None = None, parent=None):
        super().__init__(parent)
        self.callback = callback
        self.setWindowTitle(title)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.messageLabel = QLabel(message, self)
        layout.addWidget(self.messageLabel)

        self.inputBox = QLineEdit(self)
        layout.addWidget(self.inputBox)

        self.confirmButton = QPushButton("Confirm", self)
        self.confirmButton.clicked.connect(self.accept)
        layout.addWidget(self.confirmButton)

        self.setWindowFlag(Qt.WindowType.CustomizeWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

    def accept(self):
        if self.callback is not None:
            self.callback(self.inputBox.text())
        super().accept()


class ChoiceDialog(QDialog):
    choice = False

    def __init__(
        self,
        title: str,
        message: str,
        accept_callback: Callable | None = None,
        reject_callback: Callable | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.messageLabel = QLabel(message, self)
        layout.addWidget(self.messageLabel)

        button_layout = QHBoxLayout()

        # confirm and reject buttons should be put side by side
        self.confirmButton = QPushButton("Confirm", self)
        self.confirmButton.clicked.connect(self.accept)
        button_layout.addWidget(self.confirmButton)

        self.rejectButton = QPushButton("Reject", self)
        self.rejectButton.clicked.connect(self.reject)
        button_layout.addWidget(self.rejectButton)

        layout.addLayout(button_layout)

        self.setWindowFlag(Qt.WindowType.CustomizeWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        self.accept_callback = accept_callback
        self.reject_callback = reject_callback

    def accept(self):
        self.choice = True
        if self.accept_callback is not None:
            self.accept_callback()
        super().accept()

    def reject(self):
        self.choice = False
        if self.reject_callback is not None:
            self.reject_callback()
        super().reject()


class ChoiceDialogPython(ChoiceDialog):
    def __init__(
        self,
        title: str,
        message: str,
        accept_callback: Callable | None = None,
        reject_callback: Callable | None = None,
        parent=None,
    ):
        super().__init__(title, message, accept_callback, reject_callback, parent)

        self.messageLabel = Qsci.QsciScintilla()
        self.messageLabel.setLexer(Qsci.QsciLexerPython())
        self.messageLabel.setReadOnly(True)
        self.messageLabel.setText(message)

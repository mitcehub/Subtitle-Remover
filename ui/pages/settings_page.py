"""设置页：基础设置、三个模型参数、关于。"""

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                             QVBoxLayout, QWidget)

from infra.config import config, tr, VERSION
from ui.widgets.button import GhostButton
from ui.widgets.card import Card, divider
from ui.widgets.field import ComboBoxField, PathField, RangeField

PROJECT_URL = "https://github.com/mitcehub/subtitle-remover"


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner.setObjectName('SettingsScroll')
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(20, 18, 20, 28)
        layout.setSpacing(14)

        layout.addWidget(self._build_basic())
        layout.addWidget(self._build_sttn())
        layout.addWidget(self._build_e2fgvi())
        layout.addWidget(self._build_propainter())
        layout.addWidget(self._build_about())
        layout.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ---- 分组 ----

    def _build_basic(self):
        card = Card(tr["Setting"]["BasicSetting"])
        labels = list(config.interfaceTexts.keys())
        self.language_field = ComboBoxField(
            tr["SubtitleExtractorGUI"]["InterfaceLanguage"],
            config.interface, labels)
        self.save_dir_field = PathField(
            tr["Setting"]["SaveDirectory"], config.saveDirectory,
            tr["Setting"]["ChooseDirectory"],
            tr["Setting"]["SaveDirectoryDefault"])
        card.add(self.language_field)
        card.add(divider())
        card.add(self.save_dir_field)
        return card

    def _build_sttn(self):
        card = Card(tr["Setting"]["SttnSetting"])
        card.add(RangeField(tr["Setting"]["SttnMaxLoadNum"], config.sttnMaxLoadNum,
                            tr["Setting"]["SttnMaxLoadNumDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["SttnNeighborStride"], config.sttnNeighborStride,
                            tr["Setting"]["SttnNeighborStrideDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["SttnReferenceLength"], config.sttnReferenceLength,
                            tr["Setting"]["SttnReferenceLengthDesc"]))
        return card

    def _build_e2fgvi(self):
        card = Card(tr["Setting"]["E2FGVISetting"])
        card.add(RangeField(tr["Setting"]["E2FGVIMaxFrames"], config.e2fgviMaxFrames,
                            tr["Setting"]["E2FGVIMaxFramesDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["E2FGVINeighborStride"], config.e2fgviNeighborStride,
                            tr["Setting"]["E2FGVINeighborStrideDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["E2FGVIReferenceLength"], config.e2fgviReferenceLength,
                            tr["Setting"]["E2FGVIReferenceLengthDesc"]))
        return card

    def _build_propainter(self):
        card = Card(tr["Setting"]["ProPainterSetting"])
        card.add(RangeField(
            tr["Setting"]["ProPainterChunkFrames"],
            config.propainterChunkFrames,
            tr["Setting"]["ProPainterChunkFramesDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["ProPainterLocalFrames"], config.propainterLocalFrames,
                            tr["Setting"]["ProPainterLocalFramesDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["ProPainterReferenceFrames"], config.propainterReferenceFrames,
                            tr["Setting"]["ProPainterReferenceFramesDesc"]))
        card.add(divider())
        card.add(RangeField(tr["Setting"]["ProPainterStride"], config.propainterStride,
                            tr["Setting"]["ProPainterStrideDesc"]))
        return card

    def _build_about(self):
        card = Card(tr["Setting"]["AboutSetting"])

        # 项目地址
        link_row = QWidget()
        link_layout = QHBoxLayout(link_row)
        link_layout.setContentsMargins(0, 0, 0, 0)
        link_layout.setSpacing(10)

        link_text = QLabel(PROJECT_URL, objectName='FieldDesc')
        link_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        link_layout.addWidget(link_text, 1)

        open_btn = GhostButton(tr["SubtitleExtractorGUI"]["Open"])
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(PROJECT_URL)))
        link_layout.addWidget(open_btn)
        card.add(link_row)
        card.add(divider())

        # 提供反馈
        feedback_row = QWidget()
        feedback_layout = QHBoxLayout(feedback_row)
        feedback_layout.setContentsMargins(0, 0, 0, 0)
        feedback_layout.setSpacing(10)
        feedback_layout.addWidget(
            QLabel(tr["Setting"]["FeedbackDesc"], objectName='FieldDesc'), 1)
        feedback_btn = GhostButton(tr["Setting"]["FeedbackButton"])
        feedback_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(PROJECT_URL + "/issues")))
        feedback_layout.addWidget(feedback_btn)
        card.add(feedback_row)
        card.add(divider())

        # 版权
        copyright_label = QLabel(
            tr["Setting"]["CopyrightDesc"].format(VERSION), objectName='FieldDesc')
        copyright_label.setWordWrap(True)
        card.add(copyright_label)
        return card

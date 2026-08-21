from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton, QWidget

from kimix_tui.qt.composer import Composer, ComposerPad
from kimix_tui.qt.main_window import Toast
from kimix_tui.qt.transcript import MAX_TRANSCRIPT_CHARS, Transcript


def _shown(qtbot, *, max_chars: int = MAX_TRANSCRIPT_CHARS, width: int = 640, height: int = 400) -> Transcript:
    transcript = Transcript(max_chars=max_chars)
    qtbot.addWidget(transcript)
    transcript.resize(width, height)
    transcript.show()
    qtbot.waitUntil(lambda: transcript.isVisible(), timeout=5_000)
    return transcript


def _copy_pos(transcript: Transcript, row: int = 0) -> QPoint:
    rect = transcript.visualRect(transcript.model().index(row, 0))
    return QPoint(rect.right() - 10, rect.top() + 10)


def _header_pos(transcript: Transcript, row: int = 0) -> QPoint:
    rect = transcript.visualRect(transcript.model().index(row, 0))
    return QPoint(rect.left() + 24, rect.top() + 10)


def test_transcript_keeps_records_for_scrollback(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_block("user", "one")
    transcript.append_block("user", "two")
    transcript.append_block("user", "three")
    transcript.append_block("user", "four")
    assert [record.text for record in transcript.records] == ["one", "two", "three", "four"]
    assert transcript.verticalScrollBar().maximum() >= 0
    assert all(
        transcript.indexWidget(transcript.model().index(row, 0)) is None
        for row in range(len(transcript.records))
    )


def test_transcript_keeps_full_dialogue_text_when_streaming(qtbot) -> None:
    transcript = _shown(qtbot)
    first = "a" * 100
    second = "b" * 4_500
    transcript.append_stream("assistant", first)
    transcript.append_stream("assistant", second)
    assert len(transcript.records) == 1
    assert transcript.records[0].text == first + second
    user_text = "question " + ("x" * 4_500)
    transcript.append_block("user", user_text)
    assert transcript.records[-1].text == user_text


def test_transcript_keeps_full_auxiliary_text(qtbot) -> None:
    transcript = _shown(qtbot)
    tool_text = "x" * 4_500
    transcript.append_block("tool", tool_text)
    assert transcript.records[0].text == tool_text
    stream_first = "a" * 100
    stream_second = "b" * 4_500
    transcript.append_stream("thinking", stream_first)
    transcript.append_stream("thinking", stream_second)
    assert transcript.records[-1].text == stream_first + stream_second


def test_transcript_paints_records_lazily_and_bounds_memory(qtbot) -> None:
    transcript = _shown(qtbot, max_chars=32)
    transcript.append_blocks([("user", f"message-{index}-{'x' * 20}") for index in range(10)])
    assert transcript.omitted_records > 0
    assert len(transcript._strip_cache) <= 32


def test_transcript_append_blocks_keeps_full_history(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_blocks([("user", "a"), ("assistant", "b"), ("user", "c")])
    assert [record.text for record in transcript.records] == ["a", "b", "c"]


def test_non_dialogue_records_take_one_transcript_line(qtbot) -> None:
    transcript = _shown(qtbot, width=320)
    transcript.append_block("tool", "Read file\nPath: example.py\nArguments: {}")
    assert transcript._record_line_counts == [1]
    assert "▸ Read" in transcript.render_line(0).text


def test_only_explicit_copy_action_copies_dialogue_message(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_blocks(
        [
            ("system", "Session: demo"),
            ("user", "First question"),
            ("assistant", "First answer"),
            ("tool", "Read file\nPath: example.py"),
            ("tool_result", "Success\n10 lines"),
            ("user", "Second question"),
            ("assistant", "Second answer"),
        ]
    )
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(2, 0)).height() > 0)
    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    assistant_rect = transcript.visualRect(transcript.model().index(2, 0))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(assistant_rect.left() + 24, assistant_rect.top() + 36),
    )
    assert clipboard.text() == "unchanged"
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_copy_pos(transcript, 2))
    assert clipboard.text() == "First answer"
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.RightButton, pos=_header_pos(transcript, 3))
    assert clipboard.text() == "First answer"


def test_clicking_copy_action_on_system_record_copies_only_that_record(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_blocks(
        [
            ("system", "Session: demo"),
            ("user", "Question"),
            ("assistant", "Answer"),
        ]
    )
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_copy_pos(transcript, 0))
    assert QApplication.clipboard().text() == "Session: demo"


def test_drag_selects_body_text_and_ctrl_c_copies_selection(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=400)
    transcript.append_block("user", "ALPHA-SELECTABLE-TEXT OMEGA")
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    rect = transcript.visualRect(transcript.model().index(0, 0))
    start = QPoint(rect.left() + 24, rect.top() + 36)
    end = QPoint(rect.left() + 220, rect.top() + 36)
    qtbot.mousePress(transcript.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(transcript.viewport(), pos=end)
    qtbot.mouseRelease(transcript.viewport(), Qt.MouseButton.LeftButton, pos=end)
    selected = transcript.selected_text()
    assert selected
    assert selected in "ALPHA-SELECTABLE-TEXT OMEGA"
    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    transcript.setFocus()
    qtbot.keyClick(transcript, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert clipboard.text() == selected


def test_clicking_body_does_not_copy_without_selection(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_block("assistant", "Selectable assistant reply")
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    rect = transcript.visualRect(transcript.model().index(0, 0))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 24, rect.top() + 36),
    )
    assert clipboard.text() == "unchanged"
    assert transcript.selected_text() == ""
    transcript.setFocus()
    qtbot.keyClick(transcript, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert clipboard.text() == "unchanged"


def test_auxiliary_record_expands_collapses_and_copies_from_action(qtbot) -> None:
    transcript = _shown(qtbot, width=320, height=400)
    transcript.append_block("tool", "Read file\nPath: example.py")
    qtbot.waitUntil(lambda: transcript.visualRect(transcript.model().index(0, 0)).height() > 0)
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_header_pos(transcript, 0))
    assert transcript.records[0].expanded is True
    assert transcript._record_line_counts[0] > 1
    assert "Path: example.py" in transcript.visible_text()

    clipboard = QApplication.clipboard()
    clipboard.setText("unchanged")
    rect = transcript.visualRect(transcript.model().index(0, 0))
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 24, rect.top() + 40),
    )
    assert clipboard.text() == "unchanged"
    assert transcript.records[0].expanded is True

    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_copy_pos(transcript, 0))
    assert clipboard.text() == "Read file\nPath: example.py"
    qtbot.mouseClick(transcript.viewport(), Qt.MouseButton.LeftButton, pos=_header_pos(transcript, 0))
    assert transcript.records[0].expanded is False
    assert transcript._record_line_counts == [1]


def test_thinking_records_start_expanded_in_italic(qtbot) -> None:
    transcript = _shown(qtbot, width=384)
    transcript.append_block("thinking", "considering options")
    visible = transcript.visible_text()
    assert transcript.records[0].expanded is True
    assert transcript._record_line_counts[0] > 1
    assert "▾ Think" in visible
    assert "considering options" in visible
    header = transcript.render_line(0).text
    assert "considering options" not in header


def test_clicking_scrolled_message_body_does_not_copy(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    transcript.append_blocks(
        [
            item
            for turn in range(12)
            for item in (("user", f"Question {turn}"), ("assistant", f"Answer {turn}"))
        ]
    )
    qtbot.waitUntil(lambda: transcript.verticalScrollBar().maximum() > 0)
    transcript.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    target = transcript.model().index(10, 0)
    transcript.scrollTo(target)
    QApplication.clipboard().setText("unchanged")
    rect = transcript.visualRect(target)
    qtbot.mouseClick(
        transcript.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(rect.left() + 20, rect.top() + 36),
    )
    assert QApplication.clipboard().text() == "unchanged"


def test_transcript_opens_scrolled_to_latest_history(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=400)
    transcript.append_block("system", "Session: demo")
    transcript.append_blocks(
        [("user", f"history-first-{i:03d}") for i in range(50)]
        + [("assistant", "history-latest-reply")]
    )
    qtbot.waitUntil(lambda: transcript.verticalScrollBar().maximum() > 0)
    assert transcript.verticalScrollBar().value() == transcript.verticalScrollBar().maximum()
    visible = transcript.visible_text()
    assert "history-first-000" not in visible
    assert "history-latest-reply" in visible


def test_transcript_stays_put_when_user_scrolls_up(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=400)
    transcript.append_blocks([("user", f"turn-{i:03d}") for i in range(40)])
    qtbot.waitUntil(lambda: transcript.verticalScrollBar().maximum() > 0)
    transcript.verticalScrollBar().setValue(0)
    QApplication.processEvents()
    assert transcript.verticalScrollBar().value() == 0
    assert transcript.pinned_to_latest is False
    transcript.append_block("user", "new-after-scroll")
    assert transcript.verticalScrollBar().value() == 0
    visible = transcript.visible_text()
    assert "turn-000" in visible
    assert "new-after-scroll" not in visible


def test_prepend_history_preserves_the_visible_scroll_anchor(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    transcript.append_blocks([("user", f"current-{index:02d}") for index in range(30)])
    transcript.mark_history_window(0, len(transcript.records))
    mid = max(1, transcript.verticalScrollBar().maximum() // 2)
    transcript.verticalScrollBar().setValue(mid)
    QApplication.processEvents()
    assert transcript.pinned_to_latest is False
    old_scroll = transcript.verticalScrollBar().value()
    added_lines = transcript.prepend_history_blocks([("user", "older-00"), ("user", "older-01")])
    assert added_lines == 6
    assert transcript.verticalScrollBar().value() == old_scroll + added_lines * 18
    assert [record.text for record in transcript.records[:2]] == ["older-00", "older-01"]


def test_replacing_history_window_keeps_live_rows(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_blocks(
        [("system", "Session: demo"), ("user", "old-history"), ("assistant", "live")]
    )
    transcript.mark_history_window(1, 2)
    transcript.replace_history_blocks([("user", "page-history"), ("assistant", "page-reply")])
    assert [(record.kind, record.text) for record in transcript.records] == [
        ("system", "Session: demo"),
        ("user", "page-history"),
        ("assistant", "page-reply"),
        ("assistant", "live"),
    ]
    assert transcript.history_window == (1, 3)


def test_jump_to_turn_unpins_and_scrolls_immediately(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    transcript.replace_history_blocks([("user", f"q{index}", index) for index in range(20)])
    transcript.jump_to_latest()
    assert transcript.pinned_to_latest is True
    transcript.jump_to_turn(0)
    assert transcript.pinned_to_latest is False
    assert transcript.viewport_turn() == 0
    assert "q0" in transcript.visible_text()
    assert "q19" not in transcript.visible_text()


def test_jump_to_latest_pins_without_leaving_older_rows(qtbot) -> None:
    transcript = _shown(qtbot, width=640, height=220)
    transcript.replace_history_blocks([("user", f"q{index}", index) for index in range(20)])
    transcript.jump_to_turn(0)
    assert transcript.pinned_to_latest is False
    transcript.jump_to_latest()
    assert transcript.pinned_to_latest is True
    assert transcript.verticalScrollBar().value() == transcript.verticalScrollBar().maximum()
    assert "q19" in transcript.visible_text()


def test_replace_history_keeps_in_flight_stream(qtbot) -> None:
    transcript = _shown(qtbot)
    transcript.append_block("system", "Session: demo")
    transcript.mark_history_window()
    transcript.replace_history_blocks([("user", "q0", 0)])
    transcript.append_stream("assistant", "hel")
    transcript.append_stream("assistant", "lo")
    transcript.replace_history_blocks([("user", "q0", 0), ("assistant", "old", 0)])
    transcript.append_stream("assistant", "!")
    assert [record.text for record in transcript.records] == [
        "Session: demo",
        "q0",
        "old",
        "hello!",
    ]


def test_history_window_skips_fifo_trim(qtbot) -> None:
    transcript = _shown(qtbot, max_chars=32)
    transcript.mark_history_window()
    transcript.append_blocks([("user", f"message-{index}-{'x' * 20}") for index in range(10)])
    assert transcript.omitted_records == 0
    assert len(transcript.records) == 10


def test_prompt_enter_submits_without_inserting_newline(qtbot) -> None:
    prompt = Composer("Ask")
    submitted: list[str] = []
    prompt.submitted.connect(submitted.append)
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    qtbot.keyClicks(prompt, "hi")
    qtbot.keyClick(prompt, Qt.Key.Key_Return)
    assert submitted == ["hi"]
    assert prompt.text == "hi"


def test_prompt_ctrl_enter_inserts_newline_then_enter_sends(qtbot) -> None:
    prompt = Composer("Ask")
    submitted: list[str] = []
    prompt.submitted.connect(submitted.append)
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    qtbot.keyClicks(prompt, "hi")
    qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(prompt, "there")
    assert prompt.text == "hi\nthere"
    assert submitted == []
    qtbot.keyClick(prompt, Qt.Key.Key_Return)
    assert submitted == ["hi\nthere"]


def test_prompt_shift_enter_inserts_newline(qtbot) -> None:
    prompt = Composer("Ask")
    submitted: list[str] = []
    prompt.submitted.connect(submitted.append)
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    qtbot.keyClicks(prompt, "a")
    qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    qtbot.keyClicks(prompt, "b")
    assert prompt.text == "a\nb"
    assert submitted == []


def test_prompt_grows_with_lines_and_caps_height(qtbot) -> None:
    prompt = Composer("Ask")
    qtbot.addWidget(prompt)
    prompt.show()
    prompt.setFocus()
    assert prompt.height() == Composer.MIN_HEIGHT
    assert prompt.verticalScrollBar().isVisible() is False
    prompt.setPlainText("one line of prompt text")
    assert prompt.height() == Composer.MIN_HEIGHT
    assert prompt.verticalScrollBar().isVisible() is False
    for _ in range(20):
        qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert prompt.height() == Composer.MAX_HEIGHT
    assert prompt.text.count("\n") == 20
    assert prompt.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded


def test_composer_expand_emits_and_pad_sends_long_text(qtbot) -> None:
    prompt = Composer("Ask")
    opened: list[int] = []
    prompt.expand_requested.connect(lambda: opened.append(1))
    qtbot.addWidget(prompt)
    prompt.show()
    expand = prompt.findChild(QPushButton, "expand-prompt")
    assert expand is not None
    expand.click()
    assert opened == [1]
    assert prompt.height() == Composer.MIN_HEIGHT

    submitted: list[str] = []
    long_text = "line\n" * 80 + "end"
    pad = ComposerPad(long_text)
    pad.submitted.connect(submitted.append)
    qtbot.addWidget(pad)
    pad.show()
    assert pad.findChild(QWidget, "prompt-pad") is not None
    send = pad.findChild(QPushButton, "send-pad")
    assert send is not None
    send.click()
    assert submitted == [long_text]
    assert pad.sent is True


def test_composer_pad_enter_newlines_send_requires_click(qtbot) -> None:
    pad = ComposerPad("hello")
    submitted: list[str] = []
    pad.submitted.connect(submitted.append)
    qtbot.addWidget(pad)
    pad.show()
    editor = pad.findChild(QPlainTextEdit, "prompt-pad")
    assert editor is not None
    editor.setFocus()
    editor.setPlainText("hello\nworld")
    cursor = editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    assert submitted == []
    assert editor.toPlainText() == "hello\nworld\n"
    qtbot.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert submitted == []
    assert editor.toPlainText() == "hello\nworld\n"
    send = pad.findChild(QPushButton, "send-pad")
    assert send is not None
    send.click()
    assert submitted == ["hello\nworld\n"]


def test_toast_centers_at_bottom_and_auto_hides(qtbot) -> None:
    host = QWidget()
    host.resize(800, 600)
    qtbot.addWidget(host)
    host.show()
    toast = Toast(host)
    toast.show_message("AI message copied", duration_ms=80)
    assert toast.isVisible()
    mid = toast.x() + toast.width() / 2
    assert abs(mid - 400) < 24
    assert toast.y() + toast.height() > 520
    qtbot.waitUntil(lambda: not toast.isVisible(), timeout=3_000)


def test_toast_click_dismisses_before_timer(qtbot) -> None:
    host = QWidget()
    host.resize(800, 600)
    qtbot.addWidget(host)
    host.show()
    toast = Toast(host)
    toast.show_message("Copied", duration_ms=10_000)
    assert toast.isVisible()
    qtbot.mouseClick(toast, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not toast.isVisible(), timeout=3_000)

import sqlite3
import pytest
from src.feedback_store import init_db, save_feedback
from contextlib import closing


def test_persistent_feedback(tmp_path):
    path = tmp_path / 'feedback.db'
    init_db(path)
    assert path.exists()
    assert save_feedback(path, 'one', 'Thông', 4, "Ảnh mờ; 'DROP TABLE feedback;'", result={'detections': []})
    init_db(path)
    assert not save_feedback(path, 'one', 'Thông', 4, 'repeat')
    assert save_feedback(path, 'two', '', 2, 'Nhận nhầm nền')
    with closing(sqlite3.connect(path)) as conn, conn:
        rows=conn.execute('SELECT user_name,comment FROM feedback ORDER BY id').fetchall()
    assert len(rows)==2 and rows[0][0]=='Thông'
    with pytest.raises(ValueError): save_feedback(path, 'three', '', 3, '  ')
    with pytest.raises(ValueError): save_feedback(path, 'three', '', 6, 'bad')

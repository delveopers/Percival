import pytest
from percival.ingestion import NoClipsError, scan_clips
from percival.storage import safe_name

def test_scan_sorts_skips_bad_files_and_accepts_uppercase_extensions(project):
  project.video("b.mp4", frames=10); project.video("a.mp4", frames=10); project.video("E.MP4", frames=10)
  root = project.root / "data" / "samples"
  (root / "corrupt.mp4").write_bytes(b"not a video")
  (root / "empty.mp4").write_bytes(b"")
  (root / "notes.txt").write_text("x")
  (root / ".hidden.mp4").write_bytes(b"x")
  clips = scan_clips(project.cfg())
  # assert [c.clip_id for c in clips] == ["E", "a", "b"]  # works for linux well, I guess
  assert [c.clip_id for c in clips] == ["a", "b", "E"]    # works for windows well
  assert all(c.fps == 20 and c.width == 640 and c.height == 480 for c in clips)

def test_empty_and_missing_folders_raise(project):
  with pytest.raises(NoClipsError): scan_clips(project.cfg())
  import shutil
  shutil.rmtree(project.root / "data" / "samples")
  with pytest.raises(NoClipsError, match="not found"): scan_clips(project.cfg())

def test_recursive_clip_ids_cannot_collide(project):
  project.video("same.mp4", frames=10, folder=project.root / "data" / "samples" / "a")
  project.video("same.mp4", frames=10, folder=project.root / "data" / "samples" / "b")
  project.write_config({"app": {"paths": {"recursive": True}}})
  assert [c.clip_id for c in scan_clips(project.cfg())] == ["a__same", "b__same"]

def test_safe_name_keeps_plain_names_and_cleans_awkward_ones():
  assert safe_name("sample") == "sample"
  cleaned = safe_name("Hikvision bullet camera - Velibor\u2019s World (720p, h264)")
  assert all(ch.isalnum() or ch in "._-" for ch in cleaned) and cleaned != safe_name("Hikvision bullet camera - Velibor\u2019s World (720p, h265)")
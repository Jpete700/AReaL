"""Stream ASPIRE's native agentview/wrist frames to per-execution MP4s."""

import hashlib
import json
from pathlib import Path

import numpy as np


class ExecutionVideo:
    def __init__(self, env, output: Path, writer_factory=None):
        if writer_factory is None:
            import imageio.v2 as imageio

            writer_factory = imageio.get_writer
        self.writer_factory = writer_factory
        self.env = env
        self.output = output
        self.native_record = env._record_frame
        self.fps = env._control_freq / env._subsample_rate
        self.writers = {}
        self.counts = {"agentview": 0, "wrist": 0}
        self.closed = False
        self.metadata = None
        env._record_frame = self.capture
        env.enable_video_capture(True, clear=True, wrist_camera=True)

    def capture(self):
        if self.closed:
            return
        self.native_record()
        for name, buffer in (
            ("agentview", self.env._frame_buffer),
            ("wrist", self.env._wrist_frame_buffer),
        ):
            if name not in self.writers and buffer:
                filename = (
                    "video_combined.mp4"
                    if name == "agentview"
                    else "video_combined_wrist.mp4"
                )
                self.writers[name] = self.writer_factory(
                    str(self.output / filename),
                    fps=self.fps,
                    format="FFMPEG",
                    codec="libx264",
                    ffmpeg_params=["-threads", "2"],
                )
            for frame in buffer:
                self.writers[name].append_data(np.ascontiguousarray(frame))
                self.counts[name] += 1
            buffer.clear()

    def close(self):
        if self.closed:
            return self.metadata
        errors = []
        try:
            self.capture()  # Actual final state; no synthetic stepping or padding.
        except Exception as error:
            errors.append(type(error).__name__)
        self.closed = True
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception as error:
                errors.append(type(error).__name__)
        files = {}
        for name, filename in (
            ("agentview", "video_combined.mp4"),
            ("wrist", "video_combined_wrist.mp4"),
        ):
            path = self.output / filename
            if path.is_file():
                files[name] = {
                    "path": filename,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "frames": self.counts[name],
                    "bytes": path.stat().st_size,
                    "fps": self.fps,
                    "duration_seconds": self.counts[name] / self.fps,
                }
        self.metadata = {
            "schema_version": "execution-video-v1",
            "capture": "ASPIRE native _record_frame",
            "complete": not errors
            and len(files) == 2
            and min(self.counts.values()) >= 1,
            "errors": errors,
            "files": files,
            "control_hz": self.env._control_freq,
            "capture_stride": self.env._subsample_rate,
            "simulation_steps": self.env.executed_steps,
            "dimensions": [self.env._render_width, self.env._render_height],
            "presentation_padding_frames": 0,
        }
        with (self.output / "video_manifest.json").open("x") as stream:
            json.dump(self.metadata, stream, indent=2)
            stream.write("\n")
        return self.metadata

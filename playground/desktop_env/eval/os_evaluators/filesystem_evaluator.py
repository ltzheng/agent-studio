import filecmp
import grp
import logging
import os
import pwd
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from playground.desktop_env.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)


class FilesystemEvaluator(Evaluator):
    name: str = "filesystem"

    @staticmethod
    def file_content_match(path: str, expected_content: str) -> bool:
        try:
            with open(path, "r") as file:
                content = file.read()
            return content == expected_content
        except IOError:
            return False

    @staticmethod
    def file_identical(path1: str, path2: str) -> bool:
        return filecmp.cmp(path1, path2)

    @staticmethod
    def permission_match(path: str, expected_permissions: str) -> bool:
        try:
            # Compare permissions as octal
            st_mode = os.stat(path).st_mode & 0o777
            return st_mode == int(expected_permissions, 8)
        except ValueError:
            # Convert permissions to a readable format
            st_mode = os.stat(path).st_mode
            actual_permissions = stat.filemode(st_mode)
            return actual_permissions == expected_permissions
        except IOError:
            return False

    @staticmethod
    def file_metadata_match(path: str, metadata: dict) -> bool:
        """
        metadata is a dictionary of the form:
        {
            "last_modified": "2021-09-01T12:00:00",
            "creation_time": "2021-09-01T12:00:00",
            "size": 1000,
            "owner": "user",
            "group": "group"
        }
        """

        def _compare_time(file_time: float, expected_iso_time: str) -> bool:
            file_datetime = datetime.fromtimestamp(file_time)
            expected_datetime = datetime.fromisoformat(expected_iso_time)
            return file_datetime == expected_datetime

        try:
            file_stat = os.stat(path)

            for key, value in metadata.items():
                if key == "last_modified":
                    if not _compare_time(file_stat.st_mtime, value):
                        return False
                elif key == "creation_time":
                    if not _compare_time(file_stat.st_ctime, value):
                        return False
                elif key == "size":
                    if file_stat.st_size != value:
                        return False
                elif key == "owner":
                    file_owner = pwd.getpwuid(file_stat.st_uid).pw_name
                    if file_owner != value:
                        return False
                elif key == "group":
                    file_group = grp.getgrgid(file_stat.st_gid).gr_name
                    if file_group != value:
                        return False

            return True
        except IOError:
            return False

    @staticmethod
    def folder_contains_file(folder_path: str, file_name: str) -> bool:
        folder = Path(folder_path)
        return any(f.name == file_name for f in folder.iterdir() if f.is_file())

    @staticmethod
    def exists(path: str) -> bool:
        return Path(path).exists()

    def execute(
        self, steps: list[dict[str, dict[str, Any]]], response: str | None = None
    ) -> float:
        score = 1.0
        try:
            for step in steps:
                for action, params in step.items():
                    match action:
                        case "create_file":
                            file_name = params["path"]
                            if "content" in params:
                                with open(file_name, "w") as f:
                                    f.write(params["content"])
                            else:
                                open(file_name, "w").close()
                        case "mkdir":
                            dir_name = Path(params["path"])
                            dir_name.mkdir(parents=True, exist_ok=True)
                        case "rm":
                            file_name = params["path"]
                            if os.path.exists(file_name) and os.path.isfile(file_name):
                                os.remove(file_name)
                        case "rmdir":
                            dir_name = params["path"]
                            if os.path.exists(dir_name) and os.path.isdir(dir_name):
                                shutil.rmtree(dir_name)
                        case "rename":
                            old_name = params["old_name"]
                            new_name = params["new_name"]
                            os.rename(old_name, new_name)
                        case "copy":
                            src = params["src"]
                            dest = params["dest"]
                            os.system(f"cp {src} {dest}")
                        case "move":
                            src = params["src"]
                            dest = params["dest"]
                            os.system(f"mv {src} {dest}")
                        case "chmod":
                            file_name = params["path"]
                            mode: int = int(params["mode"], 8)
                            os.chmod(file_name, mode)
                        case "exists":
                            for path, exists in params.items():
                                score *= float(
                                    FilesystemEvaluator.exists(path) == exists
                                )
                        case "type_check":
                            for path, content in params.items():
                                if content == "file":
                                    score *= float(Path(path).is_file())
                                elif content == "folder":
                                    score *= float(Path(path).is_dir())
                        case "permissions_check":
                            for path, permissions in params.items():
                                score *= float(self.permission_match(path, permissions))
                        case "content_check":
                            for path, content in params.items():
                                score *= float(self.file_content_match(path, content))
                        case "metadata_check":
                            for path, metadata in params.items():
                                score *= float(self.file_metadata_match(path, metadata))
                        case _:
                            raise Exception(f"Action {action} not found")
        except Exception as e:
            logger.error(f"An error occurred in Filesystem env: {e}")
            score = 0.0

        return score

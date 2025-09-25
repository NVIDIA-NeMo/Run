# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
import re

from invoke.context import Context

from nemo_run.core.packaging.base import Packager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GitArchivePackager(Packager):
    """
    Uses `git archive <https://git-scm.com/docs/git-archive>`_ for packaging your code.

    At a high level, it works in the following way:

    #. base_path = ``git rev-parse --show-toplevel``.
    #. Optionally define a subpath as ``base_path/self.subpath`` by setting ``subpath`` attribute.
    #. ``cd base_path`` && ``git archive --format=tar.gz --output={output_file} {self.ref}:{subpath}``
    #. This extracted tar file becomes the working directory for your job.

    .. note::
        git archive will only package code committed in the specified ref.
        Any uncommitted code will not be packaged.
        We are working on adding an option to package uncommitted code but it is not ready yet.
    """

    basepath: str = ""
    #: Relative subpath in your repo to package code from.
    #: For eg, if your repo has three folders a, b and c
    #: and you specify a as the subpath, only files inside a
    #: will be packaged. In your job, the root workdir will be
    #: a/.
    subpath: str = ""

    #: Git ref to use for archiving the code.
    #: Can be a branch name or a commit ref like HEAD.
    ref: str = "HEAD"

    #: Include submodules in the archive.
    include_submodules: bool = True

    #: Include extra files in the archive which matches include_pattern
    #: This str will be included in the command as: find {include_pattern} -type f to get the list of extra files to include in the archive
    include_pattern: str | list[str] = ""

    #: Relative path to use as tar -C option - need to be consistent with include_pattern
    #: If not provided, will use git base path.
    include_pattern_relative_path: str | list[str] = ""

    check_uncommitted_changes: bool = False
    check_untracked_files: bool = False

    def _concatenate_tar_files(
        self, ctx: Context, output_file: str, files_to_concatenate: list[str]
    ):
        """Concatenate multiple uncompressed tar files into a single tar archive.

        The list should include ALL fragments to merge (base + additions).
        Creates/overwrites `output_file`.
        """
        if not files_to_concatenate:
            raise ValueError("files_to_concatenate must not be empty")

        # Quote paths for shell safety
        quoted_files = [shlex.quote(f) for f in files_to_concatenate]
        quoted_output_file = shlex.quote(output_file)

        if os.uname().sysname == "Linux":
            # Start from the first archive then append the rest, to avoid self-append issues
            first_file, *rest_files = quoted_files
            ctx.run(f"cp {first_file} {quoted_output_file}")
            if rest_files:
                ctx.run(f"tar Af {quoted_output_file} {' '.join(rest_files)}")
            # Remove all input fragments
            ctx.run(f"rm {' '.join(quoted_files)}")
        else:
            # Extract all fragments and repack once (faster than iterative extract/append)
            temp_dir = f"temp_extract_{uuid.uuid4()}"
            ctx.run(f"mkdir -p {temp_dir}")
            for file in quoted_files:
                ctx.run(f"tar xf {file} -C {temp_dir}")
            ctx.run(f"tar cf {quoted_output_file} -C {temp_dir} .")
            ctx.run(f"rm -r {temp_dir} {' '.join(quoted_files)}")

    def package(self, path: Path, job_dir: str, name: str) -> str:
        output_file = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(output_file):
            return output_file

        if self.basepath:
            path = Path(self.basepath)

        subprocess.check_call(f"cd {str(path)} && git rev-parse", shell=True)
        output = subprocess.run(
            f"cd {str(path)} && git rev-parse --show-toplevel",
            check=True,
            stdout=subprocess.PIPE,
            shell=True,
        )
        git_base_path = Path(output.stdout.splitlines()[0].decode().strip())
        git_sub_path = os.path.join(self.subpath, "")

        if self.check_uncommitted_changes:
            try:
                subprocess.check_call(
                    f"cd {shlex.quote(str(git_base_path))} && git diff-index --quiet HEAD --",
                    shell=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "Your repo has uncommitted changes. Please commit your changes or set check_uncommitted_changes to False to proceed with packaging."
                ) from e

        if self.check_untracked_files:
            untracked_files = subprocess.run(
                f"cd {shlex.quote(str(git_base_path))} && git ls-files --others --exclude-standard",
                shell=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            assert not bool(untracked_files), (
                "Your repo has untracked files. Please track your files via git or set check_untracked_files to False to proceed with packaging."
            )

        ctx = Context()
        # Build the base uncompressed archive, then separately generate all additional fragments.
        # Finally, concatenate all fragments in one pass for performance and portability.
        base_tar_tmp = f"{output_file}.tmp.base"
        git_archive_cmd = f"git archive --format=tar --output={shlex.quote(base_tar_tmp)} {self.ref}:{git_sub_path}"
        git_submodule_cmd = "git submodule foreach --recursive 'git archive --format=tar --prefix=$sm_path/ --output=$sha1.tmp HEAD'"

        with ctx.cd(git_base_path):
            ctx.run(git_archive_cmd)
            if self.include_submodules:
                ctx.run(git_submodule_cmd)
        if isinstance(self.include_pattern, str):
            self.include_pattern = [self.include_pattern]

        if isinstance(self.include_pattern_relative_path, str):
            self.include_pattern_relative_path = [self.include_pattern_relative_path]

        if len(self.include_pattern) != len(self.include_pattern_relative_path):
            raise ValueError(
                "include_pattern and include_pattern_relative_path should have the same length"
            )

        # Collect submodule tar fragments (named as <40-hex-sha1>.tmp) if any
        submodule_tmp_files: list[str] = []
        if self.include_submodules:
            for dirpath, _dirnames, filenames in os.walk(git_base_path):
                for filename in filenames:
                    if re.fullmatch(r"[0-9a-f]{40}\.tmp", filename):
                        submodule_tmp_files.append(os.path.join(dirpath, filename))

        # Generate additional fragments from include patterns and collect their paths
        additional_tmp_files: list[str] = []
        for include_pattern, include_pattern_relative_path in zip(
            self.include_pattern, self.include_pattern_relative_path
        ):
            pattern_file_id = uuid.uuid4()
            pattern_tar_file_name = f"additional_{pattern_file_id}.tmp"

            if include_pattern == "":
                continue
            include_pattern_relative_path = include_pattern_relative_path or shlex.quote(
                str(git_base_path)
            )
            relative_include_pattern = os.path.relpath(
                include_pattern, include_pattern_relative_path
            )
            pattern_tar_file_name = os.path.join(git_base_path, pattern_tar_file_name)
            include_pattern_cmd = f"find {relative_include_pattern} -type f | tar -cf {shlex.quote(pattern_tar_file_name)} -T -"

            with ctx.cd(include_pattern_relative_path):
                ctx.run(include_pattern_cmd)
            additional_tmp_files.append(pattern_tar_file_name)

        # Concatenate all fragments in one pass into {output_file}.tmp
        fragments_to_merge: list[str] = [base_tar_tmp] + submodule_tmp_files + additional_tmp_files
        with ctx.cd(git_base_path):
            self._concatenate_tar_files(ctx, f"{output_file}.tmp", fragments_to_merge)

        gzip_cmd = f"gzip -c {output_file}.tmp > {output_file}"
        rm_cmd = f"rm {output_file}.tmp"

        with ctx.cd(git_base_path):
            ctx.run(gzip_cmd)
            ctx.run(rm_cmd)

        return output_file


__all__ = ["GitArchivePackager"]

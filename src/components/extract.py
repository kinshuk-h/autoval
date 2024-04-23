import os
import glob
import zipfile

import regex
import tqdm.auto as tqdm

from . import task
from ..utils import io

STUDENT_REGEX = r"(?P<student>[\w\s]+\w)"
VERSION_REGEX = r"Version (?P<version>\d+)/(?P<file>.*\.zip)"

def make_tree(paths):
    paths = [ path.rstrip('/').split('/') for path in paths ]
    tree  = io.Record()
    for path in paths:
        tree.deepset(path, {})
    return tree

class ExtractTask(task.Task):
    def __init__(self, root_dir, students, dir_structure, assignment=3, skip_existing=False) -> None:
        super().__init__("EXTRACT", [
            self.extract_raw_sources,
            self.unpack_and_prepare_records
        ])

        self.root_dir   = root_dir
        self.raw_dir    = os.path.join(root_dir, "raw")
        self.record_dir = os.path.join(root_dir, "records")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)

        if isinstance(assignment, int):
            assignment = f"Assignment {assignment}"
        self.assignment    = assignment
        self.dir_structure = self.compile_dir_structure(dir_structure)

        self.skip_existing = skip_existing

        self.students = students

    @staticmethod
    def compile_dir_structure(dir_structure: dict[str, str|list[str]]):
        dir_structure = {
            name: r"(?:[^\/]+\/)*(?P<path>" + (path if isinstance(path, str) else '\\/'.join(path)) + r")(?:[\/])?"
            for name, path in dir_structure.items()
        }
        return {
            name: regex.compile(path, regex.UNICODE | regex.IGNORECASE)
            for name, path in dir_structure.items()
        }

    @staticmethod
    def decompile_dir_structure(dir_structure: dict[str, regex.Pattern[str]]):
        return {
            name: pattern.pattern[22:-10]
            for name, pattern in dir_structure.items()
        }

    def extract_raw_sources(self):
        """ Extract student specific submissions from downloaded submitted files directory. """

        self.print("Extracting student submissions for", self.assignment, "to", self.raw_dir)

        submitted_files = glob.glob("OneDrive*.zip", root_dir=self.root_dir)
        for file in submitted_files:
            students = {}
            self.print("Unpacking", file, "...")
            with zipfile.ZipFile(os.path.join(self.root_dir, file)) as archive:
                # Extract [student_name]/<assignment>/* to <target_dir>
                for name in archive.namelist():
                    if match := regex.search(rf"(?ui)^{STUDENT_REGEX}\/{self.assignment}\/{VERSION_REGEX}$", name):
                        student, version, file = match['student'], int(match['version']), match['file']
                        student_data = { 'file': file, 'version': version }
                        if student in students:
                            if version > students[student]['version']:
                                students[student] = student_data
                        else:
                            students[student] = student_data

                for student, data in tqdm.tqdm(students.items(), total=len(students)):
                    member = f"{student}/{self.assignment}/Version {data['version']}/{data['file']}"
                    member = archive.getinfo(member)
                    member.filename = os.path.join(student, data['file'])
                    if not self.skip_existing or not os.path.exists(os.path.join(self.raw_dir, student, data['file'])):
                        archive.extract(member, self.raw_dir)

    def select_by_dir_structure(self, paths):
        selected_paths    = []
        selected_patterns = []
        patterns          = { **self.dir_structure }
        for path in paths:
            for name, pattern in patterns.items():
                if pattern.match(path):
                    selected_paths.append(path)
                    selected_patterns.append(name)
                    del patterns[name]
                    break
        return selected_paths, selected_patterns, self.decompile_dir_structure(patterns)

    def unpack_and_prepare_records(self):
        students = self.students or os.listdir(self.raw_dir)

        self.print("Unpacking student code archives and preparing data records ...")
        for student in (pbar := tqdm.tqdm(students)):
            pbar.set_description(student)
            record = io.Record.load_if(os.path.join(self.record_dir, f"{student}.json"))
            if not self.skip_existing or record.deepget('meta.root') is None:
                record['meta'] = {
                    'name'   : student,
                    'archive': io.get_file(os.path.join(self.raw_dir, student), 'zip')
                }
                meta_root = os.path.join(self.raw_dir, student, "files")
                with zipfile.ZipFile(os.path.join(self.raw_dir, student, record['meta']['archive'])) as archive:
                    members, dest_paths, remaining_patterns = self.select_by_dir_structure(archive.namelist())
                    os.makedirs(meta_root, exist_ok=True)
                    for member, path in zip(members, dest_paths):
                        member = archive.getinfo(member)
                        member.filename = path
                        archive.extract(member, meta_root)
                    record['files'] = {
                        'found'  : dict(zip(dest_paths, members)),
                        'missing': remaining_patterns
                    }
                    record['meta']['root'] = meta_root
                record.save(os.path.join(self.record_dir, f"{student}.json"))
import os
import subprocess

import numpy

from .task import Task
from ..utils import io, common

class CheckSimilarityTask(Task):
    def __init__(self, data_dir, base_file, skip_existing):
        super().__init__("PLAGIARISM", [
            self.check_plag_with_moss,
            self.check_plag_with_outputs
        ])

        self.data_dir          = data_dir
        self.base_file         = base_file
        self.record_dir        = os.path.join(self.data_dir, "records")
        self.plag_results_dir  = os.path.join(self.data_dir, "plagiarism")

        os.makedirs(self.plag_results_dir, exist_ok=True)

        self.students_list = [ file[:-5] for file in os.listdir(self.record_dir) ]
        self.skip_existing = skip_existing

    def check_plag_with_moss(self):
        self.print("Submitting code to MOSS for plagiarism check ...")

        base_name = os.path.splitext(os.path.basename(self.base_file))[0]
        args = [ "perl", "moss", "-l", "python", "-b", self.base_file ]

        if not os.path.exists(os.path.join(self.plag_results_dir, "moss.json")):
            for student in self.students_list:
                student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))
                if (file := student_data.deepget(("meta", "code", base_name))) is not None:
                    args.append(file)

            result = subprocess.run(args, capture_output=True)
            lines = result.stdout.decode('utf-8').strip().split('\n')

            data = {
                'result': lines[-1],
                'output': lines
            }
            io.write_json(data, os.path.join(self.plag_results_dir, "moss.json"))
        else:
            data = io.read_json(os.path.join(self.plag_results_dir, "moss.json"))
        self.print("MOSS Results:", data['result'])

    def check_plag_with_outputs(self):
        self.print("Checking plagiarism based on submitted outputs ...")

        # TODO
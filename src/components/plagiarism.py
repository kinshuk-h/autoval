import os
import subprocess

import bs4
import numpy
import regex
import pandas
import requests

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

    def parse_cell(self, string):
        if match := regex.search(r"\(\d+%\)$", string):
            sim = match[0].strip('()')
            name = os.path.split(os.path.split(string[:-len(sim)].strip())[0])[-1].strip().replace('_', ' ')
            return { 'name': name, 'sim': sim }

    def parse_moss_results(self, url):
        response = requests.get(url)
        response.raise_for_status()
        page = bs4.BeautifulSoup(response.text, features="lxml")

        pair_data, table = [], page.find("table")

        for row in table("tr")[1:]:
            cells = row("td")
            pair_data.append({
                'left': {
                    'student': self.parse_cell(str(cells[0].text)),
                    'url': cells[0].find("a")['href']
                },
                'right': {
                    'student': self.parse_cell(str(cells[1].text)),
                    'url': cells[1].find("a")['href']
                },
                'lines': int(cells[2].string)
            })

        student_data = { student: {} for student in self.students_list }

        for pair in pair_data:
            student_data[pair['left']['student']['name']][pair['right']['student']['name']] = pair['left']['student']['sim']
            student_data[pair['right']['student']['name']][pair['left']['student']['name']] = pair['right']['student']['sim']

        return pair_data, student_data

    def check_plag_with_moss(self):
        self.print("Submitting code to MOSS for plagiarism check ...")

        base_name = os.path.splitext(os.path.basename(self.base_file))[0]
        args = [ "perl", "moss", "-l", "python", "-b", self.base_file ]

        if not self.skip_existing or not os.path.exists(os.path.join(self.plag_results_dir, "moss.json")):
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

        if not self.skip_existing or not os.path.exists(os.path.join(self.plag_results_dir, "moss_summary.csv")):
            pair_data, student_data = self.parse_moss_results(data['result'])
            data.update(matches=pair_data, similarity=student_data)
            io.write_json(data, os.path.join(self.plag_results_dir, "moss.json"))

            records = []
            for student, subdata in student_data.items():
                records.append({
                    'Student': student,
                    'Involved in Plag?': len(subdata) > 0,
                    'Number of Cases': len(subdata),
                    'Min. Similarity': '' if len(subdata) == 0 else min(subdata.values(), key=lambda x: int(x[:-1])),
                    'Max. Similarity': '' if len(subdata) == 0 else max(subdata.values(), key=lambda x: int(x[:-1]))
                })

            results = pandas.DataFrame.from_records(records)
            results.to_csv(os.path.join(self.plag_results_dir, "moss_summary.csv"), index=True)
        else:
            results = pandas.read_csv(os.path.join(self.plag_results_dir, "moss_summary.csv"))

        print(results)

    def check_plag_with_outputs(self):
        self.print("Checking plagiarism based on submitted outputs ...")

        # TODO
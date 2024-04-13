import os

import pandas
# from matplotlib import pyplot

from .task import Task
from ..utils import io, common

def flatten_keys(dict_obj: dict, prefix='', result=None):
    if result is None: result = {}

    for key, value in dict_obj.items():
        if isinstance(value, dict):
            flatten_keys(value, f"{key}.", result)
        else:
            result[f"{prefix}{key}"] = value
    return result

class AggregateStatisticsTask(Task):
    def __init__(self, data_dir, marks_distribution, students=None) -> None:
        super().__init__("STATS", [
            self.compute_scores_and_stats,
            self.elaborate_student_details,
            self.describe_output_stats,
        ])

        self.data_dir   = data_dir
        self.record_dir = os.path.join(self.data_dir, "records")
        self.stats_dir  = os.path.join(self.data_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)

        self.students = students
        self.students_list = [ file[:-5] for file in os.listdir(self.record_dir) ]

        self.marks_dist = marks_distribution

    def compute_scores_and_stats(self):
        self.print("Aggregating marks statistics from test results ... ")

        max_marks = sum(
            marks for test_marks_dist in self.marks_dist.values()
            for marks in test_marks_dist.values()
        )
        self.print("Maximum marks awardable:", max_marks)

        df_columns = { 'student': [] }
        df_columns.update({
            f"{suite_name}.{test_name}": []
            for suite_name in self.marks_dist for test_name in self.marks_dist[suite_name]
        })

        for student in (pbar := common.tqdm(self.students_list)):
            pbar.set_description(student)

            student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))
            df_columns['student'].append(student)

            for suite_name, test_marks_dist in self.marks_dist.items():
                for test_name, test_max_marks in test_marks_dist.items():
                    test_status = student_data.deepget(("tests", suite_name, test_name, "status"), False)
                    df_columns[f"{suite_name}.{test_name}"].append(test_max_marks if test_status else 0)

        marks_table = pandas.DataFrame(df_columns)
        marks_table['total'] = marks_table.iloc[:, 1:].sum(axis=1)
        marks_table.to_csv(os.path.join(self.stats_dir, "student_marks_data.csv"), index=False, header=True)

    def elaborate_student_details(self):
        self.print("Generating student test reports ...", ("skip" if self.students is None else ""))

        if self.students is None: return

        for student in self.students:
            student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))
            marks_awarded, marks_total = 0, 0

            self.print()
            self.print("STUDENT:", student)
            for suite_name, test_marks_dist in self.marks_dist.items():
                self.print("    TEST SUITE:", suite_name)
                tlen = max(len(name) for name in test_marks_dist)
                for test_name, test_max_marks in test_marks_dist.items():
                    marks_total += test_max_marks
                    test_data = student_data.deepget(("tests", suite_name, test_name), None)
                    if test_data is None:
                        status_string = "unevaluated"
                    elif test_data['status']:
                        status_string = "pass"
                        marks_awarded += test_max_marks
                    else:
                        status_string = test_data['reason'] or 'test assertion failed'
                        if '\n' in status_string: status_string = status_string.split('\n')[0] + " ... "
                        status_string = f"fail: {status_string}"
                    self.print("         ", f"{test_name:{tlen}}", f"({test_max_marks:2}):", status_string)
            self.print("    MARKS:", marks_awarded, "/", marks_total)

    def describe_output_stats(self):
        self.print("Generating statistics for module outputs ...")

        student_outputs = []

        for student in (pbar := common.tqdm(self.students_list)):
            pbar.set_description(student)

            student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))
            student_outputs.append({
                suite_name: student_data.deepget(("outputs", suite_name), {})
                for suite_name in self.marks_dist
            })

        for suite_name in self.marks_dist:
            suite_outputs = [ flatten_keys(output[suite_name]) for output in student_outputs ]
            if all(bool(output) == False for output in suite_outputs): continue
            print()
            self.print("Output statistics for", suite_name, "...")

            print(pandas.DataFrame.from_records(suite_outputs).describe())
import os

import pandas
# from matplotlib import pyplot

from .task import Task
from ..utils import io, common

class AggregateStatisticsTask(Task):
    def __init__(self, data_dir, marks_distribution) -> None:
        super().__init__("STATS", [
            self.compute_scores_and_stats
        ])

        self.data_dir   = data_dir
        self.record_dir = os.path.join(self.data_dir, "records")
        self.stats_dir  = os.path.join(self.data_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)

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


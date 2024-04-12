import os
import argparse

from src.utils import io
from src.components import *

def make_parser():
    parser = argparse.ArgumentParser(description="Performs evaluation of submitted assignments")
    parser.add_argument("-D", "--data-dir", default="data", help="directory for data")
    parser.add_argument('--config', default="config/ds207-assignment-3.json",
                        help="path to config file for assignment specific evaluation")
    subparsers = parser.add_subparsers(dest="action", description="action to perform")
    parser.add_argument("-S", "--skip-existing", action="store_true",
                        help="skip processing of existing results")
    parser.add_argument("-s", "--students", nargs='+', default=None, required=False,
                        help="list of student names to process")

    extract_parser = subparsers.add_parser("extract",
                                           help="perform extraction from downloaded archives for evaluation")

    code_parser = subparsers.add_parser("code", help="performs code extraction and module preparation")
    code_parser.add_argument('-C', '--invalidate-cache', action='store_true',
                             help=(
                                 'invalidate existing results and re-execute '
                                 'code preparation (ignores skip_existing)'
                             ))
    code_parser.add_argument('-t', '--templates-dir', default='templates',
                             help='directory for code templates to populate')

    test_parser = subparsers.add_parser("test", help="performs test execution on extracted code modules")
    test_parser.add_argument('-C', '--invalidate-cache', action='store_true',
                             help=(
                                 'invalidate existing results and re-execute '
                                 'test execution (ignores skip_existing)'
                             ))
    test_parser.add_argument('-t', '--test-dir', default='tests',
                             help='directory for test modules to execute')
    test_parser.add_argument('-m', '--modules', nargs='*', default=None, required=False,
                             help="specific modules to test, defaulting to all")

    stats_parser = subparsers.add_parser("stats", help="compiles test results and creates marks statistics")

    return parser, [ extract_parser, code_parser, test_parser, stats_parser ]

def main():
    parser, subparsers = make_parser()
    args = parser.parse_args()

    action = getattr(args, 'action') or 'all'
    if action == 'all':
        for subparser in subparsers:
            def_args = subparser.parse_args([])
            for arg in vars(def_args):
                if arg not in args:
                    setattr(args, arg, getattr(def_args, arg))

    if not os.path.exists(args.config):
        print("error:", "config file", args.config, "not found.")

    context = io.read_json(args.config)

    if action in ('extract', 'all'):
        extract.ExtractTask(
            args.data_dir,
            assignment    = context["assignment"],
            dir_structure = context['assignment_files'],
            skip_existing = args.skip_existing
        ).execute()

    if action in ('code', 'all'):
        code.CodeBlockPreparationTask(
            args.data_dir,
            code_filename  = context['assignment_code_file'],
            code_variables = context['assignment_variables'],
            templates_dir  = args.templates_dir, students_list = args.students,
            skip_existing  = args.skip_existing and not args.invalidate_cache
        ).execute()

    if action in ('test', 'all'):
        execute.TestExecutorTask(
            args.data_dir, args.test_dir,
            students_list=args.students, modules=args.modules,
            skip_existing=args.skip_existing and not args.invalidate_cache
        ).execute()

    if action in ('stats', 'all'):
        statistics.AggregateStatisticsTask(
            args.data_dir, context['marks_distribution']
        ).execute()

if __name__ == "__main__":
    main()

import os
import glob
import traceback

import regex

from . import task
from ..utils import io, common

VAR_PATTERN = r"(?umi)^\s*(?P<statement>{0}\s*=\s*(?P<value>{1}))"
FORMAT = {
    'int' : r"\d+",
    'bool': r"(?:True|False)",
    'str' : r"\"[^\"]+\"",
    'dict': r"(?:\{(?:\s*[\"'][^\"']+[\"']\s*:\s*[^,\}]+,)*(?:\s*[\"'][^\"']+[\"']\s*:\s*[^,\}]+)?\s*\})|(?:dict\((?:\s*(?:[a-z_]\w*)\s*=\s*[^,)]+\s*,)*(?:\s*(?:[a-z_]\w*)\s*=\s*[^,)]+\s*)?\s*\))",
    'code': r".+",
}

TEMPLATE_INSERT_PATTERN = regex.compile(
    r"(?ui)^(?P<indent>[ ]*)# >>> \{(?P<type>variable|block|segment):(?P<name>[\w.-]+)(?::(?P<mode>stmt|value))?\} <<<"
)

def block_indent(block, indent):
    """ Indents code in a given code block.

    Args:
        block (list[str]): Code block to indent.
        indent (int): Number of spaces to indent with.

    Returns:
        list[str]: Indented code block.
    """

    if indent == 0: return block
    indent = ' ' * indent
    return [ indent + line for line in block ]

def block_deindent(block):
    """ De-indents the code block to the best possible, while respecting sub-indents.

    Args:
        block (list[str]): Code block to de-indent.

    Returns:
        list[str]: De-indented code block, with preserved sub-indent.
    """

    if len(block) == 0: return block

    min_deindent = max(len(line) for line in block)
    for line in block:
        if len(line.strip()) == 0: continue
        if match := regex.search(r"(?ui)^\s+", line):
            indent = len(match[0])
        else: indent = 0
        min_deindent = min(indent, min_deindent)
    return block if min_deindent == 0 else [ line[min_deindent:] for line in block ]

def block_strip(block):
    """ Strips extra lines from a block, such as empty lines. """
    start_idx, end_idx = 0, len(block)
    while start_idx < end_idx and len(block[start_idx].strip()) == 0: start_idx += 1
    while end_idx > start_idx and len(block[end_idx-1].strip()) == 0: end_idx -= 1
    return block[start_idx:end_idx]

def validate_line_for_strip(line):
    """ Validates whether a line can be stripped from a segment. """
    return len(line.strip()) == 0 or (regex.search(r"^\s*#\s*.*", line) is not None)

def segment_strip(block, start, end):
    """ Strips off undesirable lines from a segment, such as comments and empty lines. """
    start_idx, end_idx = start, end
    while start_idx < end_idx and validate_line_for_strip(block[start_idx]): start_idx += 1
    while end_idx > start_idx and validate_line_for_strip(block[end_idx-1]): end_idx -= 1
    return start_idx, end_idx

class CodeBlockPreparationTask(task.Task):
    """ Sub-task for extracting code implementations and creating code modules. """

    def __init__(self, data_dir, code_filename, code_variables,
                templates_dir, students_list=None, skip_existing=False) -> None:
        """ Initializes the code-block preparation task.

        Args:
            data_dir (str): Directory for data (root)
            code_filename (str): Code file under student work files.
            code_variables (dict[str, str]): Variables map to parse from student code files.
            templates_dir (str): Directory containing code templates.
            students_list (list[str], optional): List of students to process results for. Defaults to all.
            skip_existing (bool, optional): If true, skips processed results in favor of the cache. Defaults to False.
        """

        super().__init__("CODE_EXTRACT", [
            self.prepare_code_blocks,
            self.create_code_from_templates
        ])

        self.data_dir      = data_dir
        self.students_list = students_list

        self.skip_existing = skip_existing

        self.code_dir = os.path.join(self.data_dir, "code")
        os.makedirs(self.code_dir, exist_ok=True)

        self.templates_dir = templates_dir
        self.record_dir    = os.path.join(self.data_dir, "records")

        if self.students_list is None:
            self.students_list = [ file[:-5] for file in os.listdir(self.record_dir) ]

        self.code_filename  = code_filename
        self.code_variables = self.compile_variables(code_variables)

    @staticmethod
    def compile_variables(code_variables):
        """ Creates regex patterns from a specified variable map.

        Args:
            code_variables (dict[str, str]): Mapping of variable to type.

        Returns:
            dict[str, dict[str, any]]: Mapping of variable to variable pattern and data.
        """

        new_code_variables = {}
        for name, dtype in code_variables.items():
            var_pattern = regex.compile(VAR_PATTERN.format(name, FORMAT[dtype]))
            new_code_variables[name] = {
                'pattern': var_pattern,
                'type'   : dtype,
            }
        return new_code_variables

    @staticmethod
    def parse_code_block(block):
        """ Parses the code segments from a code block, demarcated by the special BEGIN CODE and END CODE blocks.

        Args:
            block (list[str]): Code block to parse.

        Returns:
            tuple[str, list[dict[str, any]]]: block name and list of segment objects.
        """

        segments, segment_name, in_code_segment, start = [], "", False, 0
        # io.jprint(block)
        for index, line in enumerate(block):
            # print(f"{index:4} |", line)
            if match := regex.search(r"(?ui)^\s*#+ BEGIN CODE\s*:\s*([\w.-]+)\s*$", line):
                segment_name, start, in_code_segment = match[1], index+1, True
                if segment_name == "rnn-enc-dec-attn.attentions":
                    segment_name = "enc-dec-rnn-attn.attentions"
            elif regex.search(r"(?ui)^\s*#+\s*END CODE", line):
                in_code_segment = False
                segments.append({
                    'name': segment_name,
                    'code': segment_strip(block, start, index),
                    'reference': None
                })
        if in_code_segment:
            segments.append({
                'name': segment_name,
                'code': segment_strip(block, start, index+1),
                'reference': None
            })
        segments = [ segment for segment in segments if segment['name'] ]
        if len(segments) > 0:
            segment_names = [ segment['name'] for segment in segments ]
            block_name = segment_names[0]
            for name in segment_names[1:]:
                cursor, max_cursor = 0, min(len(block_name), len(name))
                while cursor < max_cursor and block_name[cursor] == name[cursor]: cursor += 1
                block_name = block_name[:cursor] if cursor else ''
                if block_name == '': break
            block_name = block_name.rstrip('.-')
        else: block_name = ''
        for segment in segments: segment['reference'] = block_name
        return block_name, segments

    @classmethod
    def parse_code_blocks(cls, code_blocks):
        """ Parses data in unison from multiple code blocks, and aggregates the results.
            Internally uses `parse_code_block` for each block.

        Args:
            code_blocks (list[list[str]]): List of code blocks.

        Returns:
            tuple[list[dict[str, any]], list[dict[str, any]]]: list of block and segment objects respectively.
        """
        new_code_blocks, code_segments = [], []

        for block in code_blocks:
            block_name, segments = cls.parse_code_block(block)
            if len(segments) > 0:
                code_segments.extend(segments)
                new_code_blocks.append({
                    'name': block_name,
                    'code': block,
                })

        return new_code_blocks, code_segments

    def parse_non_block_code(cls, non_block_lines, code_blocks, code_segments):
        """ Parses non-blocked code segments from left-over lines in the code.

        Args:
            non_block_lines (list[str]): List of lines not part of any block.
            code_blocks (list): List of code blocks.
            code_segments (list): List of code segments.

        Returns:
            tuple[list, list]: List of updated code blocks and segments lists.
        """

        possible_blocks, current_block, in_segment = [], [], False

        for line in non_block_lines:
            if regex.search(r"(?ui)^\s*#+ BEGIN CODE\s*:\s*([\w.-]+)\s*$", line):
                in_segment = True
            elif regex.search(r"(?ui)^\s*#+\s*END CODE", line):
                current_block.append(line)
                possible_blocks.append(block_deindent(block_strip(current_block)))
                in_segment, current_block = False, []
            if in_segment:
                current_block.append(line)

        new_code_blocks, new_code_segments = cls.parse_code_blocks(possible_blocks)
        code_blocks.extend(new_code_blocks)
        code_segments.extend(new_code_segments)

        return code_blocks, code_segments

    def parse_variables(self, code):
        """ Parses variables from given code, using the compiled variable map.

        Args:
            code (str): Raw source code to search for variables.

        Returns:
            dict[str, str]: Mapping of variables to values from the source code.
        """
        variables = {}
        for name, variable in self.code_variables.items():
            for match in variable['pattern'].findall(code):
                if variable['type'] != 'code' and len(match[1]):
                    variables[name] = match[1]
                elif match[1].lower() != 'none':
                    variables[name] = match[0]
        return variables

    def prepare_code_blocks(self):
        """ Primary sub-task: extracting code blocks and segments from student code. """

        self.print("Extracting implemented code segments and variables from code files ...")

        for student in (pbar := common.tqdm(self.students_list)):
            pbar.set_description(student)

            student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))
            code_file = os.path.join(student_data.deepget("meta.root"), self.code_filename)

            if not self.skip_existing or student_data.get('code') is None:
                code_blocks = []

                if not os.path.exists(code_file):
                    self.print("error: code implementation not found, for student:", student)

                else:
                    code, non_block_lines = "", []
                    with open(code_file, "r", encoding="utf-8") as ifile:
                        eval_block, in_eval_block = [], False
                        for line in ifile:
                            code += line
                            if regex.search(r"^## [=]{2,4} BEGIN EVALUATION PORTION$", line.rstrip()):
                                in_eval_block = True
                            elif regex.search(r"^## [=]{2,4} END EVALUATION PORTION$", line.rstrip()):
                                code_blocks.append(block_deindent(block_strip(eval_block)))
                                eval_block, in_eval_block = [], False
                            elif in_eval_block:
                                eval_block.append(line.rstrip())
                            elif not in_eval_block:
                                non_block_lines.append(line.rstrip())

                    code_blocks, code_segments = self.parse_code_blocks(code_blocks)
                    code_blocks, code_segments = self.parse_non_block_code(
                        non_block_lines, code_blocks, code_segments
                    )

                    student_data['code'] = {
                        'blocks': {
                            block['name']: block
                            for block in code_blocks
                        },
                        'segments': {
                            segment['name']: segment
                            for segment in code_segments
                        }
                    }
                    student_data['variables'] = self.parse_variables(code)
                    student_data.save(os.path.join(self.record_dir, f"{student}.json"))

    def create_code_from_templates(self):
        """ Secondary sub-task: Creating code modules from code templates with student code. """

        self.print("Creating file dumps for specified code templates ...")

        templates = glob.glob(os.path.join(self.templates_dir, "*.py"))

        for template in templates:
            filename = os.path.basename(template)
            self.print("Creating code file from template:", filename, "...")

            for student in (pbar := common.tqdm(self.students_list)):
                pbar.set_description(student)

                student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))

                os.makedirs(os.path.join(self.code_dir, student), exist_ok=True)
                student_code = os.path.join(self.code_dir, student, filename)

                try:
                    if not os.path.exists(student_code) or not self.skip_existing:
                        write_lines = []

                        with open(template, 'r', encoding='utf-8') as ifile:
                            for line in ifile:
                                line = line if line[-1] != '\n' else line[:-1]
                                if match := TEMPLATE_INSERT_PATTERN.search(line):

                                    if match['type'] == 'variable':
                                        if match['name'] not in student_data['variables']:
                                            raise ValueError(f"variable {match['name']} not found")
                                        if match['mode'] == 'stmt':
                                            var_value = f"{match['name']} = {student_data['variables'][match['name']]}"
                                        else:
                                            var_value = student_data['variables'][match['name']]
                                        var_block = var_value.split('\n')
                                        write_lines.extend(block_indent(var_block, len(match['indent'])))

                                    elif match['type'] == 'block':
                                        block = student_data.deepget(("code", "blocks", match['name'], "code"))
                                        if block is None: raise ValueError(f"block {match['name']} not found")
                                        write_lines.extend(block_indent(block, len(match['indent'])))

                                    elif match['type'] == 'segment':
                                        segment_data = student_data.deepget(("code", "segments", match['name']))
                                        if segment_data is None: raise ValueError(f"segment {match['name']} not found")
                                        block = student_data['code']['blocks'][segment_data['reference']]['code']
                                        segment = block_deindent(block[segment_data['code'][0]:segment_data['code'][1]])
                                        write_lines.extend(block_indent(segment, len(match['indent'])))

                                else:
                                    write_lines.append(line)

                        with open(student_code, 'w', encoding='utf-8') as ofile:
                            for line in write_lines: ofile.write(line + '\n')

                        student_data.deepset(f"meta.code.{filename[:-3]}", student_code)
                        student_data.save(os.path.join(self.record_dir, f"{student}.json"))
                except Exception as exc:
                    if len(self.students_list) == 1:
                        for line_set in traceback.format_exception(exc):
                            for line in line_set.split('\n'): self.print("   ", line)
                    self.print("Could not generate template", filename, "for student:", student, ":", exc)
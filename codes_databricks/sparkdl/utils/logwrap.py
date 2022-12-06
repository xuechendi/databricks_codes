# Copyright 2018 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import textwrap


def fill(text, width=100, title=None):
    """
    Fills text to fit line width, preserving paragraphs and lists.

    It first removes common leading whitespace from every line.
    Then it fills each paragraph and list item separately.
    A list item starts with "*", where the leading whitespace is preserved.
    A paragraph or a list item ends with a blank line or another list item.
    Blank lines are removed.

    :param text: input text
    :param width: line width to fit. This is for testing only.
                  We should always use the default value for consistency.
    :param title: optional title, which becomes the first line wrapped by "### ... ###".
                  If title is not None, we add one newline at the beginning and one at the end to
                  make the text blob easy to discover among log messages.
    :return: formatted text
    """
    dedented = textwrap.dedent(text)
    paragraphs = []
    lines = []

    if title is not None:
        title_line = "\n### {} ###".format(title)
        paragraphs.append(title_line)

    def fill_previous():
        if lines:
            is_list = lines[0].lstrip().startswith("*")
            indent = lines[0].find("*") if is_list else 0
            initial_indent = " " * indent
            subsequent_indent = " " * (indent + 2) if is_list else ""
            stripped_lines = [l.lstrip() for l in lines]
            filled = textwrap.fill("\n".join(stripped_lines), width=width,
                                   initial_indent=initial_indent,
                                   subsequent_indent=subsequent_indent)
            paragraphs.append(filled)
            del lines[:]
    for line in dedented.splitlines():
        stripped = line.lstrip()
        if stripped == "" or stripped.startswith("*"):
            fill_previous()
        if stripped != "":
            lines.append(line)
    fill_previous()

    if title is not None:
        paragraphs.append("")

    return "\n".join(paragraphs)

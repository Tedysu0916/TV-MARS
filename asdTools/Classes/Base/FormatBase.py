from asdTools.Classes.Base.RewriteBase import RewriteBase
import csv


class FormatBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def filter_csv_empty(self, content:list) -> list:
        """
        Filter out empty rows from the CSV content.

        Args:
            content (list): The CSV content to filter.

        Returns:
            list: The filtered content without empty rows.
        """
        res = []
        for row in content:
            isEmpty = True
            for col in row:
                if col != '':
                    isEmpty = False
                    break
            if not isEmpty:
                res.append(row)
        return res

    def read_csvLike(self, path:str, delimiter:str=',') -> list:
        """
        Read and parse the CSV-like file from the given path.

        Args:
            path (str): The path to the CSV-like file.
            delimiter (str, optional): The delimiter used in the file. Defaults to ','.

        Returns:
            list: The parsed content as a list of rows.
        """
        with open(path, 'r', encoding='utf8') as f:
            content = csv.reader(f, delimiter=delimiter)
            content = list(content)
        return content

    def sort_csv_by_col1(self, content:list, keepHead:bool=True) -> bool:
        if keepHead:
            content_head = content[0]
            content = content[1:]
        content = sorted(content, key=lambda row: row[0])
        if keepHead:
            content.insert(0, content_head)
        return content

    def save_csv(self, content:list, output_dir:str='', output_file:str='styles.csv', delimiter=',') -> str:
        """
        Sort the CSV-like content by the values in the first column.

        Args:
            content (list): The CSV-like content to sort.
            keepHead (bool, optional): Whether to keep the header row. Defaults to True.

        Returns:
            bool: True if the sorting is successful, False otherwise.
        """
        output_path = self.generate_output_path(output_dir=output_dir, output_file=output_file)
        with open(output_path, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerows(content)
        return output_path



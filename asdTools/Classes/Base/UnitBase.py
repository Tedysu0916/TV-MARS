from asdTools.Classes.Base.RewriteBase import RewriteBase


class UnitBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def convert_storage_units(self, size:float, input_unit:str="B", output_unit:str="GB"):
        """
        Convert storage units from one unit to another.

        Args:
            size (float): The size value to convert.
            input_unit (str): The input unit of the size value. Default is "B" (bytes).
            output_unit (str): The desired output unit. Default is "GB" (gigabytes).

        Returns:
            float: The converted size value in the specified output unit.
        """
        unit_level = {"b":0, "B":1, "KB":2, "MB":3, "GB":4, "TB":5}
        for i in range(unit_level[input_unit]):
            if i == 0:
                size *= 8
            else:
                size *= 1024
        for i in range(unit_level[output_unit]):
            if i == 0:
                size /= 8
            else:
                size /= 1024
        return size


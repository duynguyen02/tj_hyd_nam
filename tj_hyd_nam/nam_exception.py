class MissingColumnsException(Exception):
    def __init__(self, col: str):
        super(f"Missing columns, please provide the following column: '{col}'.")


class ColumnContainsEmptyDataException(Exception):
    def __init__(self, col: str):
        super(f"Column '{col}' contains empty data.")


class InvalidDatetimeException(Exception):
    def __init__(self):
        super(f"Invalid datetime.")


class InvalidDatetimeIntervalException(Exception):
    def __init__(self):
        super(f"Invalid datetime interval.")
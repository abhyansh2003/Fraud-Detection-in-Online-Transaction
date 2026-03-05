import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = CustomException.get_detailed_error_message(
            error_message, error_detail
        )
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error in file [{file_name}] at line [{line_no}] : {error_message}"
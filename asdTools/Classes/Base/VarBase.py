from asdTools.Classes.Base.RewriteBase import RewriteBase
import numpy as np
import operator
import random
import json
import math
import copy
import ast


class VarBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def convert_val_adaptive(self, strr:str) -> dict:  
        """
        Recursively convert values in a nested dictionary to their appropriate data types.

        Args:
            strr (str or dict or list): The input string, dictionary, or list to convert.

        Returns:
            dict: The converted dictionary with appropriate data types.
        """
        # Thanks to ChatGPT      
        if isinstance(strr, dict):
            return {self.convert_val_adaptive(k): self.convert_val_adaptive(v) for k, v in strr.items()}
        if isinstance(strr, list):
            return [self.convert_val_adaptive(e) for e in strr]
        if isinstance(strr, str):
            return self.parse_value(strr)
        return strr

    def convert_json_to_str(self, jsonn:dict) -> str:
        """
        Convert dictionary keys and values to strings.

        Args:
            jsonn (dict): The dictionary to convert.

        Returns:
            str: The converted dictionary with string keys and values.
        """
        json_res = {}
        for k, v in jsonn.items():
            json_res[str(k)] = str(v)
        return json_res

    def convert_arr_item_to_str(self, arr:np.ndarray):
        """
        Convert items in a numpy array to strings.

        Args:
            arr (np.ndarray): The numpy array to convert.

        Returns:
            list: The converted list with string items.
        """
        listt = arr.tolist()
        listt = self.convert_list_item_to_str(listt)
        return listt

    def convert_list_item_to_str(self, listt:list):
        """
        Convert items in a list to strings.

        Args:
            listt (list): The list to convert.

        Returns:
            list: The converted list with string items.
        """
        list_res = []
        for item in listt:
            if isinstance(item, list):
                list_res.append(self.convert_list_item_to_str(item))
            else:
                list_res.append(str(item))
        return list_res


    def count_in_dict(self, dictt:dict, key:str, value=1) -> bool:
        """
        Count the occurrence of a key in a dictionary.

        Args:
            dictt (dict): The dictionary to count in.
            key (str): The key to count.
            value (int): The value to increment by. Default is 1.

        Returns:
            bool: True if the key is found and count is incremented, False otherwise.
        """
        if key in dictt:
            dictt[key] += value
            return True
        else:
            dictt[key] = value
            return False

    def check_equal(self, val1, val2) -> bool:
        """
        Check if two values are equal.

        Args:
            val1: The first value.
            val2: The second value.

        Returns:
            bool: True if the values are equal, False otherwise.
        """
        if isinstance(val1, list):
            if not isinstance(val1, list):
                return False
            val1_sort = sorted(val1)
            val2_sort = sorted(val2)
            isEqual = val1_sort == val2_sort
            return isEqual

    def deepcopy(self, var):
        var_copy = copy.deepcopy(var)
        return var_copy

    def get_dict_value(self, dictt:dict, key:str, default_value):
        """
        Get the value of a key from a dictionary, or return a default value if the key does not exist.

        Args:
            dictt (dict): The dictionary to get the value from.
            key (str): The key to retrieve the value for.
            default_value: The default value to return if the key is not found.

        Returns:
            The value of the key if it exists, otherwise the default value.
        """
        try:
            if key in dictt:
                value = dictt[key]
            else:
                value = default_value
        except:
            value = default_value
        return value

    def get_list_value(self, listt:list, index:int, default_value):
        """
        Get the value at a specific index in a list, or return a default value if the index is out of range.

        Args:
            listt (list): The list to get the value from.
            index (int): The index to retrieve the value for.
            default_value: The default value to return if the index is out of range.

        Returns:
            The value at the specified index if it exists, otherwise the default value.
        """
        try:
            value = listt[index]
        except:
            value = default_value
        return value

    def json_write_pass_empty(self, jsonn:dict, key:str, value) -> bool:        
        """
        Write key-value pair to a dictionary if value is not empty.

        Args:
        - json (dict): The dictionary to write to.
        - key (str): The key of the key-value pair to write.
        - value (str or list): The value of the key-value pair to write.

        Returns:
        - bool: True if the value is not empty and has been successfully written to the dictionary, False otherwise.
        """
        # wait for improvement
        if isinstance(value, str):
            if value != "":
                jsonn[key] = value
                return True
        elif isinstance(value, list):
            if value:
                jsonn[key] = value
                return True
        return False

    def merge_dict(self, dict1:dict, dict2:dict) -> dict:
        """
        Merge two dictionaries by adding the values of common keys.

        Args:
            dict1 (dict): The first dictionary.
            dict2 (dict): The second dictionary.

        Returns:
            dict: The merged dictionary.
        """
        # Only test in int type.
        dic_res = self.deepcopy(dict1)
        for k2, v2 in dict2.items():
            v1 = self.get_dict_value(dic_res, k2, 0)
            value = self.merge_value(v1, v2)
            dic_res[k2] = value
        return dic_res

    def merge_2list_to_dcit(self, key_list, val_list, empty_value="EMPTY VALUE"):
        """
        Merge two lists into a dictionary, using the items in the first list as keys and the items in the second list as values.

        Args:
            key_list (list): The list of keys.
            val_list (list): The list of values.
            empty_value: The value to assign if the corresponding value is not present in val_list.

        Returns:
            dict: The merged dictionary.
        """
        res = {}
        for i, key in enumerate(key_list):
            value = self.get_list_value(val_list, i, empty_value)
            res[key] = value
        return res

    def merge_value(self, val1, val2):
        """
        Merge two values by adding them together.

        Args:
            val1: The first value.
            val2: The second value.

        Returns:
            The merged value.
        """
        if isinstance(val1, int):
            return val1 + val2
        elif isinstance(val1, float):
            return val1 + val2
        elif isinstance(val1, str):
            return val1 + val2
        elif isinstance(val1, list):
            return val1.extend(val2)

    def parse_value(self, value):
        """
        Parse a string value into its appropriate data type.

        Args:
            value: The value to parse.

        Returns:
            The parsed value with its appropriate data type.
        """
        # Thanks to ChatGPT
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError):
            pass
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError):
            pass
        return value

    def split_list(self, listt:list, ratios:tuple, isRandom:bool=True):
        """
        Split a list into multiple sublists based on ratios.

        Args:
            listt (list): The list to split.
            ratios (tuple): The ratios to split the list by.
            isRandom (bool): Whether to shuffle the list before splitting. Default is True.

        Returns:
            list: The list of sublists resulting from the split.
        """
        listt_copy = listt.copy()
        res = []
        if isRandom:
            random.shuffle(listt_copy)
        if isinstance(ratios, float):
            ratios = (ratios)
        start = 0
        ratio_total = 0
        for ratio in ratios:
            ratio_total += ratio
            if ratio_total >= 0.99999:
                res.append(listt_copy[start:])
                break
            else:
                split_len = math.ceil(len(listt) * ratio)
                res.append(listt_copy[start:start+split_len])
                start += split_len
        return res

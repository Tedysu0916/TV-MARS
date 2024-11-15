import winreg
import sys
import os


if __name__ == "__main__":
    try:
        # if sys.argv[4] == "True":
        #     print("delete")
        #     key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "*\\shell", 0, winreg.KEY_ALL_ACCESS)
        #     winreg.DeleteKey(key, sys.argv[2])
        #     winreg.CloseKey(key)
        # else:
        python_path = sys.argv[1].strip('"')
        sub_menu_name = sys.argv[2].strip('"')
        current_dir = sys.argv[3].strip('"')
        parent_dir = os.path.join(current_dir, "../..")
        file_path = os.path.join(parent_dir, "Tools/Image/ImageResizer_256x144.py")
        file_path = os.path.abspath(file_path)

        command = rf"{python_path} {file_path} \"%1\""
        key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "*\\shell", 0, winreg.KEY_ALL_ACCESS)
        # menu_key = winreg.CreateKey(key, "asdTools")
        sub_menu_key = winreg.CreateKey(key, sub_menu_name)
        command_key = winreg.SetValue(sub_menu_key, "command", winreg.REG_SZ, command)
        # file_type_key = winreg.CreateKey(sub_menu_key, "AppliesTo")
        # winreg.SetValue(file_type_key, ".", winreg.REG_SZ, "System.FileAssociations\image")
        # winreg.CloseKey(file_type_key)
        winreg.CloseKey(sub_menu_key)
        # winreg.CloseKey(menu_key)
        winreg.CloseKey(key)

    except Exception as e:
        print(e)
        input("Press Enter to exit.")

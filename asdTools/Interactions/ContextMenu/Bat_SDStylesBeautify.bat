@echo on
  
chcp 65001
  
set "python_path=F:\0_DATA\2_CODE\Anaconda\envs\asdTools\python.exe"
   
set "menu_name=SDStyles美化"
@REM set "menu_name=SDStylesBeautify"


set "script_dir=%~dp0"
set "script_name=%ContextMenu_SDStylesBeautify.py"
"%python_path%" "%script_dir%\%script_name%" "%python_path%" "%menu_name%" "%script_dir%"

pause
